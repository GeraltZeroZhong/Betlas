from __future__ import annotations

import json
import math
import subprocess
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from ...constants import STANDARD_AMINO_ACIDS
from ...features.extract import extract_signature
from ...features.geometry import EPS
from ...models import (
    DomainCandidate,
    ResidueRecord,
    SecondaryStructureElement,
    SheetPatch,
    StructureGeometry,
)
from ...provenance import build_run_manifest, write_json
from ..topology_diagnostics import compute_topology_diagnostics

READOUT_NAME = "BFVD viral beta-fold grammar scan"

BETA_CODES = {"E", "B"}
HELIX_CODES = {"H", "G", "I"}

AUDIT_COLUMNS = [
    "bfvd_id",
    "UniRef100",
    "taxid",
    "lineage",
    "length",
    "split_status",
    "avg_pLDDT",
    "pTM",
    "BFVD_version",
    "beta_content",
    "parse_status",
    "Betlas_top1",
    "top2",
    "top2_margin",
    "entropy",
    "sandwichness",
    "jelly_rollness",
    "closure_like_evidence",
    "mixed_topology_flag",
    "boundary_score",
    "nearest_Foldseek_known_hit",
    "hit_TM/evalue",
    "annotation_text",
]


@dataclass(frozen=True)
class BFVDScanResult:
    features_csv: Path
    diagnostics_csv: Path
    audit_csv: Path
    funnel_csv: Path
    manifest_path: Path | None
    summary: pd.DataFrame


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return "" if text.lower() in {"nan", "none", "na"} else text


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value in {"", None, ".", "?"}:
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in {"", None, ".", "?"}:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _model_stem(value: Any) -> str:
    text = Path(_clean_text(value)).name
    return text[:-4] if text.lower().endswith(".pdb") else text


def _uniref_from_model(model: str) -> str:
    return str(model).split("_", 1)[0]


def _split_status(model: str, raw: Any = "") -> str:
    raw_text = _clean_text(raw).strip().lower()
    if raw_text in {"1", "true", "yes", "split", "splitted"}:
        return "split"
    if raw_text in {"0", "false", "no", "unsplit", "not_split"}:
        return "unsplit"
    stem = _model_stem(model)
    return "split" if stem.rsplit("_", 1)[-1].isdigit() and "_" in stem else "unspecified"


def _read_table(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    table_path = Path(path)
    if not table_path.exists():
        return pd.DataFrame()
    return pd.read_csv(table_path, sep="\t", dtype=str, keep_default_na=False, low_memory=False)


def _standardize_metadata(metadata_tsv: str | Path | None) -> pd.DataFrame:
    df = _read_table(metadata_tsv)
    if df.empty:
        return pd.DataFrame(columns=["bfvd_id", "UniRef100", "model"])
    if "model" not in {column.lower() for column in df.columns} and metadata_tsv is not None:
        df = pd.read_csv(
            metadata_tsv,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            low_memory=False,
            header=None,
            names=["UniRef100", "model", "avg_pLDDT", "pTM", "splitted", "BFVD_version"],
        )
    df = df.copy()
    rename = {}
    for column in df.columns:
        lower = column.lower()
        if lower == "uniref100":
            rename[column] = "UniRef100"
        elif lower == "model":
            rename[column] = "model"
        elif lower in {"avg_plddt", "avgplddt"}:
            rename[column] = "avg_pLDDT"
        elif lower == "ptm":
            rename[column] = "pTM"
        elif lower in {"splitted", "split", "split_status"}:
            rename[column] = "splitted"
        elif lower == "version":
            rename[column] = "BFVD_version"
    df = df.rename(columns=rename)
    if "model" not in df.columns:
        raise ValueError(f"BFVD metadata must contain a model column: {metadata_tsv}")
    df["bfvd_id"] = df["model"].map(_model_stem)
    if "UniRef100" not in df.columns:
        df["UniRef100"] = df["bfvd_id"].map(_uniref_from_model)
    if "BFVD_version" not in df.columns:
        df["BFVD_version"] = ""
    if "splitted" not in df.columns:
        df["splitted"] = ""
    if "avg_pLDDT" not in df.columns:
        df["avg_pLDDT"] = ""
    if "pTM" not in df.columns:
        df["pTM"] = ""
    return df.drop_duplicates("bfvd_id", keep="first")


def _standardize_taxonomy(taxonomy_tsv: str | Path | None) -> pd.DataFrame:
    df = _read_table(taxonomy_tsv)
    if df.empty:
        return pd.DataFrame(columns=["bfvd_id", "taxid", "lineage", "annotation_text"])
    if "model" not in {column.lower() for column in df.columns} and taxonomy_tsv is not None:
        df = pd.read_csv(
            taxonomy_tsv,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            low_memory=False,
            header=None,
            names=["model", "taxid", "rank", "scientific_name", "lineage"],
        )
    df = df.copy()
    rename = {}
    for column in df.columns:
        lower = column.lower().replace(" ", "_")
        if lower == "model":
            rename[column] = "model"
        elif lower in {"taxid", "tax_id"}:
            rename[column] = "taxid"
        elif lower in {"scientific_name", "scientificname"}:
            rename[column] = "scientific_name"
        elif lower == "lineage":
            rename[column] = "lineage"
        elif lower == "rank":
            rename[column] = "rank"
    df = df.rename(columns=rename)
    if "model" not in df.columns:
        return pd.DataFrame(columns=["bfvd_id", "taxid", "lineage", "annotation_text"])
    df["bfvd_id"] = df["model"].map(_model_stem)
    df["UniRef100"] = df["bfvd_id"].map(_uniref_from_model)
    if "taxid" not in df.columns:
        df["taxid"] = ""
    if "lineage" not in df.columns:
        df["lineage"] = ""
    if "scientific_name" not in df.columns:
        df["scientific_name"] = ""
    if "rank" not in df.columns:
        df["rank"] = ""
    df["annotation_text"] = df["scientific_name"]
    return df[
        ["bfvd_id", "UniRef100", "taxid", "rank", "scientific_name", "lineage", "annotation_text"]
    ].drop_duplicates("UniRef100", keep="first")


def _metadata_qc_mask(
    df: pd.DataFrame,
    *,
    min_avg_plddt: float,
    min_ptm: float,
    allow_split: bool,
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    plddt = pd.to_numeric(df.get("avg_pLDDT", ""), errors="coerce")
    ptm = pd.to_numeric(df.get("pTM", ""), errors="coerce")
    split_ok = pd.Series(True, index=df.index)
    if not allow_split:
        split_status = df.apply(lambda row: _split_status(row.get("bfvd_id", ""), row.get("splitted", "")), axis=1)
        split_ok = split_status.ne("split")
    return plddt.ge(min_avg_plddt).fillna(False) & ptm.ge(min_ptm).fillna(False) & split_ok


def _discover_records(
    *,
    bfvd_dir: Path,
    metadata: pd.DataFrame,
    taxonomy: pd.DataFrame,
    min_avg_plddt: float,
    min_ptm: float,
    allow_split: bool,
    limit: int | None,
    sample_seed: int | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    pdb_paths = sorted(bfvd_dir.glob("*.pdb"))
    path_df = pd.DataFrame({"bfvd_id": [path.stem for path in pdb_paths], "pdb_path": [str(path) for path in pdb_paths]})
    metrics = {
        "bfvd_pdb_files": int(len(path_df)),
        "metadata_rows": int(len(metadata)),
        "metadata_qc_pass_rows": 0,
        "candidate_rows_before_limit": 0,
    }
    if metadata.empty:
        records = path_df.copy()
        records["UniRef100"] = records["bfvd_id"].map(_uniref_from_model)
        records["model"] = records["bfvd_id"]
        records["avg_pLDDT"] = ""
        records["pTM"] = ""
        records["splitted"] = ""
        records["BFVD_version"] = ""
    else:
        qc = _metadata_qc_mask(
            metadata,
            min_avg_plddt=min_avg_plddt,
            min_ptm=min_ptm,
            allow_split=allow_split,
        )
        metrics["metadata_qc_pass_rows"] = int(qc.sum())
        records = metadata.loc[qc].merge(path_df, on="bfvd_id", how="inner")
    if not taxonomy.empty and "UniRef100" in records.columns:
        records = records.merge(
            taxonomy.drop(columns=["bfvd_id"], errors="ignore"),
            on="UniRef100",
            how="left",
        )
    else:
        records["taxid"] = ""
        records["lineage"] = ""
        records["annotation_text"] = ""
    metrics["candidate_rows_before_limit"] = int(len(records))
    if limit is not None and limit >= 0 and len(records) > limit:
        if sample_seed is None:
            records = records.head(limit).copy()
        else:
            records = records.sample(n=limit, random_state=sample_seed).sort_values("bfvd_id").copy()
    return records.reset_index(drop=True), metrics


def _parse_ca_records(pdb_path: Path, *, chain_id: str | None = None) -> tuple[str, list[ResidueRecord], list[float]]:
    by_chain: dict[str, list[tuple[ResidueRecord, float]]] = {}
    seen: set[tuple[str, int, str]] = set()
    with pdb_path.open("rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip().upper()
            if atom_name != "CA":
                continue
            residue_name = line[17:20].strip().upper()
            if residue_name not in STANDARD_AMINO_ACIDS:
                continue
            chain = line[21].strip() or "A"
            if chain_id is not None and chain != chain_id:
                continue
            try:
                auth_seq_id = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            insertion_code = line[26].strip()
            key = (chain, auth_seq_id, insertion_code)
            if key in seen:
                continue
            seen.add(key)
            plddt = _safe_float(line[60:66], default=math.nan)
            residue = ResidueRecord(
                pdb_id=pdb_path.stem.lower(),
                chain_id=chain,
                auth_seq_id=auth_seq_id,
                label_seq_id=None,
                insertion_code=insertion_code,
                residue_name=residue_name,
                coord_ca=(x, y, z),
            )
            by_chain.setdefault(chain, []).append((residue, plddt))
    if not by_chain:
        return chain_id or "A", [], []
    selected_chain = chain_id or max(by_chain, key=lambda value: len(by_chain[value]))
    selected = sorted(by_chain.get(selected_chain, []), key=lambda item: (item[0].auth_seq_id, item[0].insertion_code))
    return selected_chain, [item[0] for item in selected], [item[1] for item in selected]


def _with_pdb_header(path: Path) -> Path:
    first = path.open("rt", encoding="utf-8", errors="replace").readline()
    if first.startswith("HEADER"):
        return path
    tmp = tempfile.NamedTemporaryFile("wt", suffix=".pdb", delete=False, encoding="utf-8")
    tmp_path = Path(tmp.name)
    with tmp:
        tmp.write(f"HEADER    BFVD {path.stem}\n")
        with path.open("rt", encoding="utf-8", errors="replace") as source:
            for line in source:
                tmp.write(line)
    return tmp_path


def _dssp_cache_path(cache_dir: Path, model_id: str) -> Path:
    shard = model_id[:2] if len(model_id) >= 2 else "xx"
    return cache_dir / shard / f"{model_id}.dssp"


def _find_dssp_cache(cache_dir: Path | None, model_id: str) -> Path | None:
    if cache_dir is None:
        return None
    sharded = _dssp_cache_path(cache_dir, model_id)
    flat = cache_dir / f"{model_id}.dssp"
    if sharded.exists():
        return sharded
    if flat.exists():
        return flat
    return None


def _parse_dssp_file(path: Path) -> dict[tuple[str, int, str], str]:
    out: dict[tuple[str, int, str], str] = {}
    in_table = False
    known_codes = set("HBEGITSP")
    with path.open("rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.lstrip().startswith("#  RESIDUE"):
                in_table = True
                continue
            if not in_table or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            seq_id = _safe_int(parts[1], default=-10**9)
            if seq_id == -10**9:
                continue
            chain = parts[2].strip() or "A"
            ss = "-"
            if len(parts) >= 5 and len(parts[4]) == 1 and parts[4] in known_codes:
                ss = parts[4]
            out[(chain, seq_id, "")] = ss
    return out


def _run_mkdssp_to_file(pdb_path: Path, out_path: Path, *, dssp_bin: str) -> None:
    dssp_input = _with_pdb_header(pdb_path)
    remove_tmp = dssp_input != pdb_path
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [dssp_bin, "--output-format", "dssp", str(dssp_input), str(out_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            fallback = subprocess.run(
                [dssp_bin, str(dssp_input), str(out_path)],
                check=False,
                capture_output=True,
                text=True,
            )
            if fallback.returncode != 0:
                message = (fallback.stderr or result.stderr or fallback.stdout or result.stdout).strip()
                raise RuntimeError(message or f"{dssp_bin} failed")
    finally:
        if remove_tmp:
            try:
                dssp_input.unlink()
            except OSError:
                pass


def _dssp_codes(
    pdb_path: Path,
    *,
    dssp_bin: str,
    cache_dir: Path | None,
    write_cache: bool,
) -> tuple[dict[tuple[str, int, str], str], bool]:
    model_id = pdb_path.stem
    cached = _find_dssp_cache(cache_dir, model_id)
    if cached is not None:
        return _parse_dssp_file(cached), True
    if cache_dir is not None and write_cache:
        cache_path = _dssp_cache_path(cache_dir, model_id)
        _run_mkdssp_to_file(pdb_path, cache_path, dssp_bin=dssp_bin)
        return _parse_dssp_file(cache_path), False
    with tempfile.NamedTemporaryFile(suffix=".dssp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _run_mkdssp_to_file(pdb_path, tmp_path, dssp_bin=dssp_bin)
        return _parse_dssp_file(tmp_path), False
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _group_sse_segments(
    residues: list[ResidueRecord],
    ss_by_key: dict[tuple[str, int, str], str],
    *,
    codes: set[str],
    element_type: str,
    min_len: int,
) -> list[SecondaryStructureElement]:
    segments: list[SecondaryStructureElement] = []
    current: list[int] = []
    previous_seq: int | None = None
    for index, residue in enumerate(residues):
        key = (residue.chain_id, residue.auth_seq_id, residue.insertion_code)
        is_member = ss_by_key.get(key, "-") in codes
        contiguous = previous_seq is None or residue.auth_seq_id <= previous_seq + 1
        if is_member and (not current or contiguous):
            current.append(index)
        else:
            if len(current) >= min_len:
                segments.append(_make_segment(segments, residues, current, element_type))
            current = [index] if is_member else []
        previous_seq = residue.auth_seq_id
    if len(current) >= min_len:
        segments.append(_make_segment(segments, residues, current, element_type))
    return segments


def _make_segment(
    existing: list[SecondaryStructureElement],
    residues: list[ResidueRecord],
    indices: list[int],
    element_type: str,
) -> SecondaryStructureElement:
    prefix = "B" if element_type == "strand" else "H"
    first = residues[indices[0]]
    last = residues[indices[-1]]
    return SecondaryStructureElement(
        element_id=f"{prefix}{len(existing) + 1:03d}",
        element_type=element_type,
        chain_id=first.chain_id,
        start_auth_seq_id=first.auth_seq_id,
        end_auth_seq_id=last.auth_seq_id,
        residue_indices=tuple(indices),
        sheet_id="" if element_type != "strand" else "pending",
        sheet_range_id=str(len(existing) + 1),
    )


def _segment_coords(segment: SecondaryStructureElement, residues: list[ResidueRecord]) -> np.ndarray:
    return np.array([residues[index].coord_ca for index in segment.residue_indices], dtype=float)


def _segment_axis(coords: np.ndarray) -> np.ndarray:
    if len(coords) < 2:
        return np.array([1.0, 0.0, 0.0])
    axis = coords[-1] - coords[0]
    norm = float(np.linalg.norm(axis))
    if norm < EPS:
        return np.array([1.0, 0.0, 0.0])
    return axis / norm


def _same_sheet_pair(left: np.ndarray, right: np.ndarray) -> bool:
    if len(left) < 2 or len(right) < 2:
        return False
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    close = distances <= 6.8
    close_count = int(close.sum())
    min_close = 2 if min(len(left), len(right)) >= 3 else 1
    if close_count < min_close:
        return False
    nearest_left = np.min(distances, axis=1)
    nearest_right = np.min(distances, axis=0)
    return float(np.mean(nearest_left <= 7.2) + np.mean(nearest_right <= 7.2)) >= 0.8


def _infer_sheet_segments(
    beta_segments: list[SecondaryStructureElement],
    residues: list[ResidueRecord],
) -> list[SecondaryStructureElement]:
    n = len(beta_segments)
    if n == 0:
        return []
    parent = list(range(n))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    coords = [_segment_coords(segment, residues) for segment in beta_segments]
    for i in range(n):
        for j in range(i + 1, n):
            if _same_sheet_pair(coords[i], coords[j]):
                union(i, j)

    components: dict[int, list[int]] = {}
    for index in range(n):
        components.setdefault(find(index), []).append(index)
    ordered_components = sorted(
        components.values(),
        key=lambda indices: min(beta_segments[index].start_auth_seq_id for index in indices),
    )
    sheet_by_index: dict[int, str] = {}
    for sheet_number, indices in enumerate(ordered_components, start=1):
        sheet_id = f"S{sheet_number:03d}"
        for index in indices:
            sheet_by_index[index] = sheet_id

    updated: list[SecondaryStructureElement] = []
    previous_by_sheet: dict[str, int] = {}
    for index, segment in enumerate(beta_segments):
        sheet_id = sheet_by_index[index]
        previous_index = previous_by_sheet.get(sheet_id)
        sense = ""
        if previous_index is not None:
            dot = float(np.dot(_segment_axis(coords[previous_index]), _segment_axis(coords[index])))
            sense = "parallel" if dot >= 0.0 else "anti-parallel"
        previous_by_sheet[sheet_id] = index
        updated.append(
            SecondaryStructureElement(
                element_id=segment.element_id,
                element_type=segment.element_type,
                chain_id=segment.chain_id,
                start_auth_seq_id=segment.start_auth_seq_id,
                end_auth_seq_id=segment.end_auth_seq_id,
                residue_indices=segment.residue_indices,
                sheet_id=sheet_id,
                sheet_range_id=segment.sheet_range_id,
                sense_to_previous=sense,
            )
        )
    return updated


def build_bfvd_structure_geometry(
    pdb_path: Path,
    *,
    metadata: dict[str, Any],
    dssp_bin: str,
    dssp_cache_dir: str | Path | None = None,
    write_dssp_cache: bool = False,
) -> tuple[StructureGeometry, dict[str, Any]]:
    chain_id, residues, plddt_values = _parse_ca_records(pdb_path)
    bfvd_id = _model_stem(metadata.get("bfvd_id", pdb_path.stem))
    domain = DomainCandidate(
        record_id=bfvd_id,
        pdb_id=bfvd_id.lower(),
        chain_id=chain_id,
        domain_id=bfvd_id,
        residue_ranges=(
            f"{residues[0].auth_seq_id}-{residues[-1].auth_seq_id}:{chain_id}" if residues else ""
        ),
        fold_label_final="",
        evidence_level="BFVD_predicted_structure",
        label_source_primary="BFVD",
        label_source_supporting="ColabFold predicted structure; not a CATH domain label",
        cath_status="not_CATH_labeled",
        qc_status="external_predicted_structure_unreviewed",
    )
    warnings: list[str] = []
    if not residues:
        warnings.append("no_ca_residues")
        return (
            StructureGeometry(domain=domain, residues=tuple(), beta_segments=tuple(), helices=tuple(), sheet_patches=tuple(), warnings=tuple(warnings)),
            {"bfvd_structure_parse_ok": 0, "bfvd_avg_plddt_from_pdb": math.nan},
        )
    try:
        ss_by_key, dssp_cache_hit = _dssp_codes(
            pdb_path,
            dssp_bin=dssp_bin,
            cache_dir=Path(dssp_cache_dir) if dssp_cache_dir else None,
            write_cache=write_dssp_cache,
        )
    except Exception as exc:
        ss_by_key = {}
        dssp_cache_hit = False
        warnings.append(f"dssp_failed:{type(exc).__name__}")
    if not ss_by_key:
        warnings.append("no_dssp_secondary_structure")

    beta_segments = _group_sse_segments(
        residues,
        ss_by_key,
        codes=BETA_CODES,
        element_type="strand",
        min_len=2,
    )
    beta_segments = _infer_sheet_segments(beta_segments, residues)
    helices = _group_sse_segments(
        residues,
        ss_by_key,
        codes=HELIX_CODES,
        element_type="helix",
        min_len=3,
    )
    if not beta_segments:
        warnings.append("no_dssp_beta_segments")
    by_sheet: dict[str, list[str]] = {}
    for segment in beta_segments:
        by_sheet.setdefault(segment.sheet_id or "unknown", []).append(segment.element_id)
    sheet_patches = tuple(
        SheetPatch(sheet_id=sheet_id, strand_ids=tuple(strand_ids))
        for sheet_id, strand_ids in sorted(by_sheet.items())
    )
    plddt_arr = np.asarray(plddt_values, dtype=float)
    plddt_arr = plddt_arr[np.isfinite(plddt_arr)]
    extra = {
        "bfvd_structure_parse_ok": 1,
        "bfvd_avg_plddt_from_pdb": float(np.mean(plddt_arr)) if len(plddt_arr) else math.nan,
        "bfvd_chain_id": chain_id,
        "bfvd_dssp_cache_hit": int(dssp_cache_hit),
    }
    return (
        StructureGeometry(
            domain=domain,
            residues=tuple(residues),
            beta_segments=tuple(beta_segments),
            helices=tuple(helices),
            sheet_patches=sheet_patches,
            warnings=tuple(warnings),
        ),
        extra,
    )


def _terminal_low_confidence_tail(values: list[float], *, threshold: float = 50.0) -> tuple[int, float]:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return 0, 0.0
    leading = 0
    for value in clean:
        if value < threshold:
            leading += 1
        else:
            break
    trailing = 0
    for value in reversed(clean):
        if value < threshold:
            trailing += 1
        else:
            break
    length = max(leading, trailing)
    return length, length / max(1, len(clean))


def _extract_one(
    record: dict[str, Any],
    *,
    dssp_bin: str,
    dssp_cache_dir: str | Path | None = None,
    write_dssp_cache: bool = False,
) -> dict[str, Any]:
    pdb_path = Path(str(record["pdb_path"]))
    plddt_values: list[float] = []
    try:
        _chain, _residues, plddt_values = _parse_ca_records(pdb_path)
        geometry, bfvd_extra = build_bfvd_structure_geometry(
            pdb_path,
            metadata=record,
            dssp_bin=dssp_bin,
            dssp_cache_dir=dssp_cache_dir,
            write_dssp_cache=write_dssp_cache,
        )
        signature = extract_signature(geometry)
        row = signature.to_row()
        row["cz_error"] = ""
        row.update(bfvd_extra)
    except Exception as exc:
        bfvd_id = _model_stem(record.get("bfvd_id", pdb_path.stem))
        row = DomainCandidate(
            record_id=bfvd_id,
            pdb_id=bfvd_id.lower(),
            chain_id="",
            domain_id=bfvd_id,
            residue_ranges="",
            fold_label_final="",
            evidence_level="BFVD_predicted_structure",
            label_source_primary="BFVD",
        ).to_dict()
        row.update(
            {
                "cz_parse_ok": 0,
                "cz_error": f"{type(exc).__name__}: {exc}",
                "cz_warnings": "bfvd_scan_exception",
                "bfvd_structure_parse_ok": 0,
            }
        )
    tail_len, tail_fraction = _terminal_low_confidence_tail(plddt_values)
    bfvd_id = _model_stem(record.get("bfvd_id", pdb_path.stem))
    row.update(
        {
            "bfvd_id": bfvd_id,
            "UniRef100": _clean_text(record.get("UniRef100", _uniref_from_model(bfvd_id))),
            "taxid": _clean_text(record.get("taxid", "")),
            "lineage": _clean_text(record.get("lineage", "")),
            "split_status": _split_status(bfvd_id, record.get("splitted", "")),
            "avg_pLDDT": _clean_text(record.get("avg_pLDDT", "")),
            "pTM": _clean_text(record.get("pTM", "")),
            "BFVD_version": _clean_text(record.get("BFVD_version", record.get("version", ""))),
            "annotation_text": _clean_text(record.get("annotation_text", "")),
            "source_pdb_path": str(pdb_path),
            "source_pdb_size": int(pdb_path.stat().st_size) if pdb_path.exists() else 0,
            "length": int(_safe_float(row.get("cz_residue_count", 0), default=0)),
            "bfvd_low_conf_terminal_tail_length": tail_len,
            "bfvd_low_conf_terminal_tail_fraction": tail_fraction,
        }
    )
    return row


def _worker_extract(
    record: dict[str, Any],
    dssp_bin: str,
    dssp_cache_dir: str | Path | None,
    write_dssp_cache: bool,
) -> dict[str, Any]:
    return _extract_one(
        record,
        dssp_bin=dssp_bin,
        dssp_cache_dir=dssp_cache_dir,
        write_dssp_cache=write_dssp_cache,
    )


def _extract_parallel_bounded(
    records: list[dict[str, Any]],
    *,
    workers: int,
    max_pending_tasks: int | None,
    dssp_bin: str,
    dssp_cache_dir: str | Path | None,
    write_dssp_cache: bool,
) -> list[dict[str, Any]]:
    if not records:
        return []
    pending_limit = max(workers, int(max_pending_tasks or workers * 4))
    pending_limit = min(pending_limit, len(records))
    rows: list[dict[str, Any]] = []
    next_index = 0
    pending: set[Future[dict[str, Any]]] = set()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        with tqdm(total=len(records), desc="Scanning BFVD") as progress:
            while next_index < len(records) and len(pending) < pending_limit:
                pending.add(
                    pool.submit(
                        _worker_extract,
                        records[next_index],
                        dssp_bin,
                        dssp_cache_dir,
                        write_dssp_cache,
                    )
                )
                next_index += 1
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    rows.append(future.result())
                    progress.update(1)
                while next_index < len(records) and len(pending) < pending_limit:
                    pending.add(
                        pool.submit(
                            _worker_extract,
                            records[next_index],
                            dssp_bin,
                            dssp_cache_dir,
                            write_dssp_cache,
                        )
                    )
                    next_index += 1
    return rows


def _annotate_qc(
    features: pd.DataFrame,
    *,
    min_avg_plddt: float,
    min_ptm: float,
    min_length: int,
    max_length: int,
    max_domain_like_length: int,
    min_beta_strands: int,
    min_beta_content: float,
    max_pca_elongation: float,
    max_low_confidence_tail_fraction: float,
    allow_split: bool,
) -> pd.DataFrame:
    df = features.copy()
    avg_plddt = pd.to_numeric(df.get("avg_pLDDT", df.get("bfvd_avg_plddt_from_pdb", "")), errors="coerce")
    avg_plddt = avg_plddt.fillna(pd.to_numeric(df.get("bfvd_avg_plddt_from_pdb", ""), errors="coerce"))
    ptm = pd.to_numeric(df.get("pTM", ""), errors="coerce")
    length = pd.to_numeric(df.get("length", df.get("cz_residue_count", 0)), errors="coerce").fillna(0)
    beta_strands = pd.to_numeric(df.get("cz_beta_strand_count", 0), errors="coerce").fillna(0)
    beta_content = pd.to_numeric(df.get("cz_beta_residue_fraction", 0), errors="coerce").fillna(0)
    pca_elongation = pd.to_numeric(df.get("cz_pca_elongation", 999), errors="coerce").fillna(999)
    tail_fraction = pd.to_numeric(df.get("bfvd_low_conf_terminal_tail_fraction", 0), errors="coerce").fillna(0)
    split_status = df.get("split_status", pd.Series(["unspecified"] * len(df), index=df.index)).astype(str)
    parse_error = df.get("cz_error", pd.Series([""] * len(df), index=df.index)).astype(str)
    structure_parse_ok = pd.to_numeric(df.get("bfvd_structure_parse_ok", 0), errors="coerce").fillna(0).astype(bool)

    metadata_qc = avg_plddt.ge(min_avg_plddt) & ptm.ge(min_ptm)
    if not allow_split:
        metadata_qc = metadata_qc & split_status.ne("split")
    length_qc = length.ge(min_length) & length.le(max_length)
    beta_eligible = (
        structure_parse_ok
        & parse_error.eq("")
        & beta_strands.ge(min_beta_strands)
        & beta_content.ge(min_beta_content)
    )
    compact = (
        beta_eligible
        & length.ge(min_length)
        & length.le(max_domain_like_length)
        & pca_elongation.le(max_pca_elongation)
        & tail_fraction.le(max_low_confidence_tail_fraction)
    )
    high_confidence = metadata_qc & length_qc & compact
    df["bfvd_metadata_qc_pass"] = metadata_qc.astype(int)
    df["bfvd_length_qc_pass"] = length_qc.astype(int)
    df["bfvd_beta_rich_eligible"] = beta_eligible.astype(int)
    df["bfvd_domain_like_compact_candidate"] = compact.astype(int)
    df["bfvd_high_confidence_readout_set"] = high_confidence.astype(int)

    statuses: list[str] = []
    flag_rows: list[str] = []
    for index in df.index:
        flags: list[str] = []
        if parse_error.loc[index]:
            statuses.append("error")
            flags.append("parse_error")
        elif not structure_parse_ok.loc[index]:
            statuses.append("structure_parse_failed")
            flags.append("structure_parse_failed")
        elif beta_strands.loc[index] <= 0:
            statuses.append("no_beta_segments")
            flags.append("no_beta_segments")
        elif not beta_eligible.loc[index]:
            statuses.append("not_beta_rich")
            flags.append("not_beta_rich")
        elif not compact.loc[index]:
            statuses.append("beta_rich_qc_flagged")
        else:
            statuses.append("eligible_domain_like_beta_rich")
        if avg_plddt.loc[index] < min_avg_plddt:
            flags.append("low_avg_pLDDT")
        if ptm.loc[index] < min_ptm:
            flags.append("low_pTM")
        if length.loc[index] < min_length:
            flags.append("short_model")
        if length.loc[index] > max_length:
            flags.append("long_model")
        if length.loc[index] > max_domain_like_length:
            flags.append("multi_domain_suspected_length")
        if pca_elongation.loc[index] > max_pca_elongation:
            flags.append("elongated_noncompact")
        if tail_fraction.loc[index] > max_low_confidence_tail_fraction:
            flags.append("long_low_confidence_terminal_tail")
        if split_status.loc[index] == "split":
            flags.append("split_sequence")
        flag_rows.append(";".join(dict.fromkeys(flags)))
    df["parse_status"] = statuses
    df["bfvd_quality_flags"] = flag_rows
    return df


def _standardize_foldseek_context(path: str | Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=["bfvd_id", "nearest_Foldseek_known_hit", "hit_TM/evalue", "annotation_text_foldseek"])
    sep = "\t" if str(path).endswith((".tsv", ".tab")) else ","
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, low_memory=False)
    rename = {}
    for column in df.columns:
        lower = column.lower()
        if lower in {"bfvd_id", "query", "query_id", "model"}:
            rename[column] = "bfvd_id"
        elif lower in {"target", "target_id", "nearest_foldseek_known_hit", "hit"}:
            rename[column] = "nearest_Foldseek_known_hit"
        elif lower in {"tm", "tmscore", "tm_score", "hit_tm"}:
            rename[column] = "hit_TM"
        elif lower in {"evalue", "e_value", "hit_evalue"}:
            rename[column] = "hit_evalue"
        elif lower in {"annotation", "description", "target_description", "annotation_text"}:
            rename[column] = "annotation_text_foldseek"
    df = df.rename(columns=rename)
    if "bfvd_id" not in df.columns:
        return pd.DataFrame(columns=["bfvd_id", "nearest_Foldseek_known_hit", "hit_TM/evalue", "annotation_text_foldseek"])
    df["bfvd_id"] = df["bfvd_id"].map(_model_stem)
    if "nearest_Foldseek_known_hit" not in df.columns:
        df["nearest_Foldseek_known_hit"] = ""
    if "hit_TM" not in df.columns:
        df["hit_TM"] = ""
    if "hit_evalue" not in df.columns:
        df["hit_evalue"] = ""
    if "annotation_text_foldseek" not in df.columns:
        df["annotation_text_foldseek"] = ""
    df["hit_TM/evalue"] = df["hit_TM"].astype(str) + "/" + df["hit_evalue"].astype(str)
    return df[
        [
            "bfvd_id",
            "nearest_Foldseek_known_hit",
            "hit_TM/evalue",
            "annotation_text_foldseek",
        ]
    ].drop_duplicates("bfvd_id", keep="first")


def _review_bucket(row: pd.Series) -> str:
    jelly = _safe_float(row.get("jelly_rollness", 0.0), default=0.0)
    boundary = _safe_float(row.get("boundary_score", 0.0), default=0.0)
    closure = _safe_float(row.get("closure_like_evidence", 0.0), default=0.0)
    mixed = _safe_int(row.get("mixed_topology_flag", 0), default=0)
    top1 = _clean_text(row.get("Betlas_top1", ""))
    known = _clean_text(row.get("nearest_Foldseek_known_hit", ""))
    if jelly >= 0.55 and boundary >= 0.45:
        return "high_jelly_rollness_high_ambiguity"
    if closure >= 0.55 and top1 not in {"beta_barrel", "tim_like_beta_alpha_barrel"}:
        return "closure_like_evidence_outside_canonical_barrel_call"
    if mixed:
        return "mixed_topology_review"
    if not known and max(jelly, closure, _safe_float(row.get("sandwichness", 0.0), default=0.0)) >= 0.55:
        return "weak_known_context_clear_local_grammar"
    return "lower_priority"


def _build_audit_table(
    features: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    foldseek_context: str | Path | None,
) -> pd.DataFrame:
    merged = features.merge(diagnostics, on="record_id", how="left", suffixes=("", "_diagnostic"))
    context = _standardize_foldseek_context(foldseek_context)
    if not context.empty:
        merged = merged.merge(context, on="bfvd_id", how="left")
    else:
        merged["nearest_Foldseek_known_hit"] = ""
        merged["hit_TM/evalue"] = ""
        merged["annotation_text_foldseek"] = ""
    annotation = merged.get("annotation_text_foldseek", "").astype(str)
    base_annotation = merged.get("annotation_text", "").astype(str)
    merged["annotation_text_audit"] = annotation.where(annotation.ne(""), base_annotation)
    audit = pd.DataFrame(
        {
            "bfvd_id": merged.get("bfvd_id", ""),
            "UniRef100": merged.get("UniRef100", ""),
            "taxid": merged.get("taxid", ""),
            "lineage": merged.get("lineage", ""),
            "length": pd.to_numeric(merged.get("length", 0), errors="coerce").fillna(0).astype(int),
            "split_status": merged.get("split_status", ""),
            "avg_pLDDT": merged.get("avg_pLDDT", merged.get("bfvd_avg_plddt_from_pdb", "")),
            "pTM": merged.get("pTM", ""),
            "BFVD_version": merged.get("BFVD_version", ""),
            "beta_content": pd.to_numeric(merged.get("cz_beta_residue_fraction", 0), errors="coerce").fillna(0.0),
            "parse_status": merged.get("parse_status", ""),
            "Betlas_top1": merged.get("cz_rule_top1_label", ""),
            "top2": merged.get("cz_rule_top2_label", ""),
            "top2_margin": pd.to_numeric(merged.get("cz_rule_probability_margin", 0), errors="coerce").fillna(0.0),
            "entropy": pd.to_numeric(merged.get("cz_rule_probability_entropy", 0), errors="coerce").fillna(0.0),
            "sandwichness": pd.to_numeric(merged.get("cz_sandwichness", 0), errors="coerce").fillna(0.0),
            "jelly_rollness": pd.to_numeric(merged.get("cz_jelly_rollness", 0), errors="coerce").fillna(0.0),
            "closure_like_evidence": pd.to_numeric(merged.get("cz_barrel_likeness", 0), errors="coerce").fillna(0.0),
            "mixed_topology_flag": pd.to_numeric(merged.get("cz_mixed_topology_flag", 0), errors="coerce").fillna(0).astype(int),
            "boundary_score": pd.to_numeric(merged.get("cz_topology_ambiguity_score", 0), errors="coerce").fillna(0.0),
            "nearest_Foldseek_known_hit": merged.get("nearest_Foldseek_known_hit", ""),
            "hit_TM/evalue": merged.get("hit_TM/evalue", ""),
            "annotation_text": merged["annotation_text_audit"],
            "bfvd_quality_flags": merged.get("bfvd_quality_flags", ""),
            "bfvd_high_confidence_readout_set": pd.to_numeric(
                merged.get("bfvd_high_confidence_readout_set", 0), errors="coerce"
            ).fillna(0).astype(int),
        }
    )
    audit["review_queue_bucket"] = audit.apply(_review_bucket, axis=1)
    audit["review_priority_score"] = (
        0.35 * audit["boundary_score"].astype(float)
        + 0.25 * audit["jelly_rollness"].astype(float)
        + 0.20 * audit["closure_like_evidence"].astype(float)
        + 0.10 * audit["sandwichness"].astype(float)
        + 0.10 * audit["mixed_topology_flag"].astype(float)
    )
    priority_columns = [*AUDIT_COLUMNS, "bfvd_quality_flags", "bfvd_high_confidence_readout_set", "review_queue_bucket", "review_priority_score"]
    return audit[priority_columns].sort_values(
        ["bfvd_high_confidence_readout_set", "review_priority_score", "bfvd_id"],
        ascending=[False, False, True],
    )


def _write_funnel(
    path: Path,
    *,
    discover_metrics: dict[str, int],
    features: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        ("BFVD PDB files in local archive", discover_metrics.get("bfvd_pdb_files", 0)),
        ("BFVD metadata rows", discover_metrics.get("metadata_rows", 0)),
        ("Metadata QC-pass rows", discover_metrics.get("metadata_qc_pass_rows", 0)),
        ("PDB-backed candidates before limit", discover_metrics.get("candidate_rows_before_limit", 0)),
        ("Scanned rows", len(features)),
        ("Structure parse-ok rows", int(pd.to_numeric(features.get("bfvd_structure_parse_ok", 0), errors="coerce").fillna(0).sum())),
        ("Beta-rich eligible rows", int(pd.to_numeric(features.get("bfvd_beta_rich_eligible", 0), errors="coerce").fillna(0).sum())),
        (
            "Domain-like compact candidates",
            int(pd.to_numeric(features.get("bfvd_domain_like_compact_candidate", 0), errors="coerce").fillna(0).sum()),
        ),
        (
            "High-confidence Betlas readout set",
            int(pd.to_numeric(features.get("bfvd_high_confidence_readout_set", 0), errors="coerce").fillna(0).sum()),
        ),
    ]
    funnel = pd.DataFrame(rows, columns=["stage", "n"])
    funnel.to_csv(path, index=False)
    return funnel


def run_bfvd_scan(
    *,
    bfvd_dir: str | Path = "data/external/betlas_beta/BFVD",
    metadata_tsv: str | Path | None = None,
    taxonomy_tsv: str | Path | None = None,
    foldseek_context: str | Path | None = None,
    out_dir: str | Path = "runs/readouts/bfvd_viral_beta_fold_scan",
    limit: int | None = None,
    sample_seed: int | None = None,
    workers: int = 1,
    max_pending_tasks: int | None = None,
    dssp_bin: str = "mkdssp",
    dssp_cache_dir: str | Path | None = None,
    write_dssp_cache: bool = False,
    min_avg_plddt: float = 70.0,
    min_ptm: float = 0.45,
    min_length: int = 40,
    max_length: int = 1500,
    max_domain_like_length: int = 700,
    min_beta_strands: int = 4,
    min_beta_content: float = 0.18,
    max_pca_elongation: float = 5.0,
    max_low_confidence_tail_fraction: float = 0.25,
    allow_split: bool = False,
    write_manifest: bool = True,
) -> BFVDScanResult:
    bfvd_dir = Path(bfvd_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_csv = out_dir / "bfvd_features.csv"
    diagnostics_csv = out_dir / "bfvd_topology_diagnostics.csv"
    audit_csv = out_dir / "bfvd_viral_beta_fold_topology_audit.csv"
    funnel_csv = out_dir / "bfvd_scan_funnel.csv"
    manifest_path = out_dir / "bfvd_scan_manifest.json"

    metadata = _standardize_metadata(metadata_tsv)
    taxonomy = _standardize_taxonomy(taxonomy_tsv)
    records, discover_metrics = _discover_records(
        bfvd_dir=bfvd_dir,
        metadata=metadata,
        taxonomy=taxonomy,
        min_avg_plddt=min_avg_plddt,
        min_ptm=min_ptm,
        allow_split=allow_split,
        limit=limit,
        sample_seed=sample_seed,
    )
    start = time.perf_counter()
    record_dicts = records.to_dict(orient="records")
    feature_rows: list[dict[str, Any]] = []
    if workers <= 1:
        for record in tqdm(record_dicts, desc="Scanning BFVD"):
            feature_rows.append(
                _extract_one(
                    record,
                    dssp_bin=dssp_bin,
                    dssp_cache_dir=dssp_cache_dir,
                    write_dssp_cache=write_dssp_cache,
                )
            )
    else:
        feature_rows = _extract_parallel_bounded(
            record_dicts,
            workers=workers,
            max_pending_tasks=max_pending_tasks,
            dssp_bin=dssp_bin,
            dssp_cache_dir=dssp_cache_dir,
            write_dssp_cache=write_dssp_cache,
        )
    elapsed_seconds = time.perf_counter() - start
    features = pd.DataFrame(feature_rows)
    if not features.empty:
        order = {str(row["bfvd_id"]): index for index, row in enumerate(record_dicts)}
        features["_order"] = features["bfvd_id"].astype(str).map(order)
        features = features.sort_values("_order").drop(columns=["_order"])
    features = _annotate_qc(
        features,
        min_avg_plddt=min_avg_plddt,
        min_ptm=min_ptm,
        min_length=min_length,
        max_length=max_length,
        max_domain_like_length=max_domain_like_length,
        min_beta_strands=min_beta_strands,
        min_beta_content=min_beta_content,
        max_pca_elongation=max_pca_elongation,
        max_low_confidence_tail_fraction=max_low_confidence_tail_fraction,
        allow_split=allow_split,
    )
    features.to_csv(features_csv, index=False)

    diagnostics = compute_topology_diagnostics(features, predictions=None, model=None, k_neighbors=12)
    diagnostics.to_csv(diagnostics_csv, index=False)
    audit = _build_audit_table(features, diagnostics, foldseek_context=foldseek_context)
    audit.to_csv(audit_csv, index=False)
    funnel = _write_funnel(funnel_csv, discover_metrics=discover_metrics, features=features)

    rows_per_second = len(features) / elapsed_seconds if elapsed_seconds > 0 else 0.0
    estimate_all = discover_metrics.get("bfvd_pdb_files", 0) / rows_per_second if rows_per_second > 0 else math.nan
    estimate_qc = discover_metrics.get("candidate_rows_before_limit", 0) / rows_per_second if rows_per_second > 0 else math.nan
    summary = pd.DataFrame(
        [
            {"metric": "scanned_rows", "value": len(features)},
            {"metric": "elapsed_seconds", "value": round(elapsed_seconds, 3)},
            {"metric": "rows_per_second", "value": round(rows_per_second, 4)},
            {"metric": "estimated_all_pdb_seconds_at_this_throughput", "value": round(estimate_all, 1) if math.isfinite(estimate_all) else ""},
            {"metric": "estimated_qc_candidate_seconds_at_this_throughput", "value": round(estimate_qc, 1) if math.isfinite(estimate_qc) else ""},
            {
                "metric": "high_confidence_readout_rows",
                "value": int(pd.to_numeric(features.get("bfvd_high_confidence_readout_set", 0), errors="coerce").fillna(0).sum()),
            },
        ]
    )

    written_manifest_path: Path | None = None
    if write_manifest:
        inputs = {}
        if metadata_tsv is not None:
            inputs["metadata_tsv"] = Path(metadata_tsv)
        if taxonomy_tsv is not None:
            inputs["taxonomy_tsv"] = Path(taxonomy_tsv)
        if foldseek_context is not None:
            inputs["foldseek_context"] = Path(foldseek_context)
        metrics = {
            **discover_metrics,
            "scanned_rows": int(len(features)),
            "elapsed_seconds": float(elapsed_seconds),
            "rows_per_second": float(rows_per_second),
            "estimated_all_pdb_seconds_at_this_throughput": float(estimate_all) if math.isfinite(estimate_all) else None,
            "estimated_qc_candidate_seconds_at_this_throughput": float(estimate_qc) if math.isfinite(estimate_qc) else None,
            "structure_parse_ok_rows": int(pd.to_numeric(features.get("bfvd_structure_parse_ok", 0), errors="coerce").fillna(0).sum()),
            "beta_rich_eligible_rows": int(pd.to_numeric(features.get("bfvd_beta_rich_eligible", 0), errors="coerce").fillna(0).sum()),
            "domain_like_compact_candidate_rows": int(
                pd.to_numeric(features.get("bfvd_domain_like_compact_candidate", 0), errors="coerce").fillna(0).sum()
            ),
            "high_confidence_readout_rows": int(
                pd.to_numeric(features.get("bfvd_high_confidence_readout_set", 0), errors="coerce").fillna(0).sum()
            ),
        }
        manifest = build_run_manifest(
            command="betlas readout bfvd-viral-beta-fold-scan",
            parameters={
                "limit": limit,
                "sample_seed": sample_seed,
                "workers": workers,
                "max_pending_tasks": max_pending_tasks,
                "dssp_bin": dssp_bin,
                "dssp_cache_dir": str(dssp_cache_dir or ""),
                "write_dssp_cache": write_dssp_cache,
                "min_avg_plddt": min_avg_plddt,
                "min_ptm": min_ptm,
                "min_length": min_length,
                "max_length": max_length,
                "max_domain_like_length": max_domain_like_length,
                "min_beta_strands": min_beta_strands,
                "min_beta_content": min_beta_content,
                "max_pca_elongation": max_pca_elongation,
                "max_low_confidence_tail_fraction": max_low_confidence_tail_fraction,
                "allow_split": allow_split,
            },
            inputs=inputs,
            outputs={
                "features_csv": features_csv,
                "diagnostics_csv": diagnostics_csv,
                "audit_csv": audit_csv,
                "funnel_csv": funnel_csv,
            },
            metrics=metrics,
            extra={
                "readout": READOUT_NAME,
                "bfvd_dir": str(bfvd_dir),
                "audit_contract_columns": AUDIT_COLUMNS,
                "funnel": json.loads(funnel.to_json(orient="records")),
                "foldseek_context_loaded": foldseek_context is not None and Path(foldseek_context).exists(),
                "interpretation_guardrail": (
                    "BFVD rows are review targets from predicted viral structures, "
                    "not CATH-equivalent ground-truth fold labels."
                ),
            },
        )
        written_manifest_path = write_json(manifest_path, manifest)

    return BFVDScanResult(
        features_csv=features_csv,
        diagnostics_csv=diagnostics_csv,
        audit_csv=audit_csv,
        funnel_csv=funnel_csv,
        manifest_path=written_manifest_path,
        summary=summary,
    )
