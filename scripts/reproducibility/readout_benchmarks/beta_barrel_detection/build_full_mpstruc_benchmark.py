#!/usr/bin/env python
"""Build the full-MPStruc beta-barrel detection benchmark cohort."""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from Bio.PDB import MMCIFIO, MMCIFParser, Select
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Biopython is required for benchmark construction.") from exc


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT_DIR = REPO_ROOT / "data/readouts/beta_barrel_detection/full_mpstruc_767_neg800"
DEFAULT_EXTERNAL_DATA_ROOT = Path("data/external/betlas_benchmark_inputs")
DEFAULT_BETA_ROOT = DEFAULT_EXTERNAL_DATA_ROOT / "mpstruc_beta_barrels"
DEFAULT_BETA_XML = DEFAULT_BETA_ROOT.parent / "Mpstrucis.txt"
DEFAULT_BETA_CLASSIFICATION = DEFAULT_BETA_ROOT / "metadata/entry_classification.csv"
DEFAULT_CATH_CANDIDATES = (
    DEFAULT_EXTERNAL_DATA_ROOT
    / "cath_pisces_easy_negatives/metadata/easy_negative_candidates.csv"
)
DEFAULT_CATH_CHAIN_ROOT = (
    DEFAULT_EXTERNAL_DATA_ROOT / "cath_pisces_easy_negatives/chains"
)

MPSTRUC_ALPHA_URL = "https://blanco.biomol.uci.edu/mpstruc/listAll/mpstrucAlphaHlxTblXml"
RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb}.cif"
RCSB_CIF_GZ_URL = "https://files.rcsb.org/download/{pdb}.cif.gz"

PDB_ID_RE = re.compile(r"^[0-9A-Za-z]{4}$")
TAG_RE = re.compile(r"<([A-Za-z0-9_]+)>(.*?)</\1>")


@dataclass(frozen=True)
class MpstrucRecord:
    pdb_code: str
    mpstruc_name: str
    group_name: str
    subgroup_name: str
    species: str
    resolution: str
    source_type: str
    master_pdb_code: str
    parent_pdb_code: str


@dataclass(frozen=True)
class CathRecord:
    pdb_id: str
    pisces_chain: str
    resolved_chain: str
    method: str
    chain_length: int
    resolution: float | None
    r_value: float | None
    cath_dominant_class: int
    cath_dominant_class_name: str
    cath_dominant_class_fraction: float
    cath_dominant_architecture: int
    cath_dominant_topology: int
    cath_dominant_superfamily: int
    cath_domain_ids: str


class ChainOnlySelect(Select):
    def __init__(self, target_chain: str, model_id: int = 0):
        self.target_chain = target_chain
        self.model_id = model_id

    def accept_model(self, model: Any) -> int:
        return 1 if int(model.id) == int(self.model_id) else 0

    def accept_chain(self, chain: Any) -> int:
        return 1 if str(chain.id) == self.target_chain else 0


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is not None:
        names = fieldnames
    else:
        names = []
        for row in rows:
            for key in row:
                if key not in names:
                    names.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def file_state(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() else 0,
        "mtime": path.stat().st_mtime if path.exists() else None,
    }


def safe_part(value: object) -> str:
    text = str(value).strip()
    out = [char.lower() if char.isalnum() else "_" for char in text]
    return "".join(out).strip("_") or "blank"


def safe_chain_filename(chain_id: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]", "_", chain_id) or "CHAIN"


def clean_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tag_value(line: str, tag: str) -> str | None:
    match = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", line)
    if not match:
        return None
    return clean_text(match.group(1))


def normalize_pdb_code(value: str) -> str:
    text = value.strip().upper()
    return text if PDB_ID_RE.match(text) else ""


def parse_mpstruc_records(path: Path, *, include_related: bool = True) -> list[MpstrucRecord]:
    records: list[MpstrucRecord] = []
    current_group = ""
    current_subgroup = ""
    current_protein: dict[str, Any] | None = None
    current_member: dict[str, Any] | None = None
    related_owner: dict[str, Any] | None = None
    in_member = False

    def make_record(source: dict[str, Any], source_type: str, parent: dict[str, Any] | None, pdb_code: str | None = None) -> None:
        code = normalize_pdb_code(pdb_code or str(source.get("pdbCode", "")))
        if not code:
            return
        master = normalize_pdb_code(str((parent or source).get("pdbCode", "")))
        parent_code = normalize_pdb_code(str(source.get("pdbCode", ""))) or code
        records.append(
            MpstrucRecord(
                pdb_code=code,
                mpstruc_name=clean_text(str(source.get("name") or (parent or {}).get("name", ""))),
                group_name=current_group,
                subgroup_name=current_subgroup,
                species=clean_text(str(source.get("species") or (parent or {}).get("species", ""))),
                resolution=clean_text(str(source.get("resolution") or (parent or {}).get("resolution", ""))),
                source_type=source_type,
                master_pdb_code=master,
                parent_pdb_code=parent_code,
            )
        )

    def flush_member() -> None:
        nonlocal current_member
        if current_member is None or current_protein is None:
            current_member = None
            return
        make_record(current_member, "member", current_protein)
        if include_related:
            for related in current_member.get("_related", []):
                make_record(current_member, "related_member", current_protein, related)
        current_member = None

    def flush_protein() -> None:
        nonlocal current_protein
        flush_member()
        if current_protein is None:
            return
        make_record(current_protein, "master", current_protein)
        if include_related:
            for related in current_protein.get("_related", []):
                make_record(current_protein, "related_master", current_protein, related)
        current_protein = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "<group>":
                flush_protein()
                current_group = ""
                continue
            if line == "<subgroup>":
                flush_protein()
                current_subgroup = ""
                continue
            if line == "</subgroup>":
                flush_protein()
                continue
            if line == "<protein>":
                if not in_member:
                    flush_protein()
                    current_protein = {"_related": []}
                continue
            if line == "</protein>":
                if not in_member:
                    flush_protein()
                continue
            if line == "<memberProtein>":
                flush_member()
                current_member = {"_related": []}
                in_member = True
                continue
            if line == "</memberProtein>":
                flush_member()
                in_member = False
                continue
            if line == "<relatedPdbEntries>":
                related_owner = current_member if current_member is not None else current_protein
                if related_owner is not None:
                    related_owner.setdefault("_related", [])
                continue
            if line == "</relatedPdbEntries>":
                related_owner = None
                continue

            if related_owner is not None:
                value = tag_value(line, "pdbCode")
                if value:
                    related_owner.setdefault("_related", []).append(value)
                continue

            matched = TAG_RE.search(line)
            if not matched:
                continue
            tag, value = matched.group(1), clean_text(matched.group(2))
            if tag == "name" and current_protein is None and current_member is None:
                if value.startswith("TRANSMEMBRANE PROTEINS:"):
                    current_group = value
                elif current_group:
                    current_subgroup = value
                continue
            target = current_member if current_member is not None else current_protein
            if target is not None:
                target[tag] = value
    flush_protein()
    return records


def deduplicate_mpstruc_records(records: list[MpstrucRecord]) -> list[MpstrucRecord]:
    priority = {"master": 0, "member": 1, "related_master": 2, "related_member": 3}
    by_pdb: dict[str, MpstrucRecord] = {}
    for record in records:
        old = by_pdb.get(record.pdb_code)
        if old is None or priority.get(record.source_type, 99) < priority.get(old.source_type, 99):
            by_pdb[record.pdb_code] = record
    return [by_pdb[pdb] for pdb in sorted(by_pdb)]


def http_get(url: str, *, timeout: int, retries: int, backoff: float) -> bytes:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Betlas benchmark builder"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except Exception as exc:  # pragma: no cover - network dependent
            last = exc
            if attempt < retries:
                time.sleep(backoff * (2**attempt))
    raise RuntimeError(f"Failed to fetch {url}: {last}")


def maybe_gzip(data: bytes) -> bytes:
    return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data


def download_text(url: str, path: Path, *, timeout: int, retries: int, backoff: float, force: bool) -> Path:
    if path.exists() and path.stat().st_size > 0 and not force:
        return path
    data = http_get(url, timeout=timeout, retries=retries, backoff=backoff)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


def download_cif(pdb_id: str, out_dir: Path, *, timeout: int, retries: int, backoff: float) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{pdb_id.lower()}.cif"
    if out.exists() and out.stat().st_size > 0:
        return out
    for url, compressed in [
        (RCSB_CIF_URL.format(pdb=pdb_id.lower()), False),
        (RCSB_CIF_GZ_URL.format(pdb=pdb_id.lower()), True),
    ]:
        try:
            data = http_get(url, timeout=timeout, retries=retries, backoff=backoff)
            if compressed:
                data = maybe_gzip(data)
            if not data.lstrip().startswith(b"data_"):
                continue
            tmp = out.with_suffix(".cif.tmp")
            tmp.write_bytes(data)
            tmp.replace(out)
            return out
        except Exception:
            continue
    return None


def protein_residue_count(chain: Any) -> int:
    count = 0
    for residue in chain.get_residues():
        hetfield = residue.id[0] if isinstance(residue.id, tuple) else ""
        if str(hetfield).strip():
            continue
        if residue.has_id("CA") or residue.has_id("N") or residue.has_id("C"):
            count += 1
    return count


def largest_protein_chain(cif_path: Path, pdb_id: str, min_residues: int = 50) -> tuple[str, int] | None:
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, str(cif_path))
    except Exception:
        return None
    model = next(structure.get_models(), None)
    if model is None:
        return None
    chains = []
    for chain in model:
        n_res = protein_residue_count(chain)
        if n_res >= min_residues:
            chains.append((str(chain.id), n_res))
    if not chains:
        return None
    chains.sort(key=lambda item: (-item[1], item[0]))
    return chains[0]


def extract_chain_cif(full_cif: Path, pdb_id: str, chain_id: str, out_path: Path) -> bool:
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, str(full_cif))
        model = next(structure.get_models(), None)
        if model is None:
            return False
        chain_ids = [str(chain.id) for chain in model]
        target = chain_id if chain_id in chain_ids else chain_ids[0] if len(chain_ids) == 1 else ""
        if not target:
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        io = MMCIFIO()
        io.set_structure(structure)
        io.save(str(out_path), select=ChainOnlySelect(target, model_id=int(model.id)))
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def link_or_copy(src: Path, dst: Path, *, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        dst.symlink_to(src.resolve())


def locate_positive_entry(beta_root: Path, pdb_id: str) -> Path:
    path = beta_root / "entries" / f"{pdb_id.upper()}.cif"
    if path.exists():
        return path
    lower = beta_root / "entries" / f"{pdb_id.lower()}.cif"
    if lower.exists():
        return lower
    raise FileNotFoundError(f"Missing MPStruc beta-barrel entry CIF for {pdb_id}: {path}")


def read_beta_positives(path: Path, beta_root: Path, positive_dir: Path, link_mode: str) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in sorted(rows, key=lambda item: str(item["pdb_code"]).upper()):
        pdb_id = normalize_pdb_code(str(row["pdb_code"]))
        if not pdb_id or pdb_id in seen:
            continue
        seen.add(pdb_id)
        source = locate_positive_entry(beta_root, pdb_id)
        filename = f"{pdb_id}.cif"
        linked = positive_dir / filename
        link_or_copy(source, linked, mode=link_mode)
        chain = str(row["candidate_auth_asym_id"]).strip()
        label_chain = str(row["candidate_label_asym_id"]).strip()
        out.append(
            {
                "record_id": f"cbdpos_mpstruc_{safe_part(pdb_id)}_{safe_part(chain)}",
                "filename": filename,
                "pdb_id": pdb_id,
                "selected_chain_id": chain,
                "label_chain_id": label_chain,
                "fallback_chain_id": chain,
                "source_panel": "mpstruc_beta_barrel_full",
                "source_detail": row["class_label"],
                "source_path": str(source),
                "structure_path": str(linked),
                "y_true": 1,
            }
        )
    return out


def parse_float(value: object) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed


def read_cath_candidates(path: Path) -> list[CathRecord]:
    records: list[CathRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                records.append(
                    CathRecord(
                        pdb_id=normalize_pdb_code(row["pdb_id"]),
                        pisces_chain=str(row["pisces_chain"]),
                        resolved_chain=str(row["resolved_chain"]),
                        method=str(row["method"]),
                        chain_length=int(float(row["chain_length"])),
                        resolution=parse_float(row.get("resolution")),
                        r_value=parse_float(row.get("r_value")),
                        cath_dominant_class=int(row["cath_dominant_class"]),
                        cath_dominant_class_name=str(row["cath_dominant_class_name"]),
                        cath_dominant_class_fraction=float(row["cath_dominant_class_fraction"]),
                        cath_dominant_architecture=int(row["cath_dominant_architecture"]),
                        cath_dominant_topology=int(row["cath_dominant_topology"]),
                        cath_dominant_superfamily=int(row["cath_dominant_superfamily"]),
                        cath_domain_ids=str(row["cath_domain_ids"]),
                    )
                )
            except (KeyError, ValueError):
                continue
    return [record for record in records if record.pdb_id]


def cath_sort_key(record: CathRecord) -> tuple[float, float, float, int, str, str]:
    return (
        record.resolution if record.resolution is not None else 999.0,
        record.r_value if record.r_value is not None else 999.0,
        -record.cath_dominant_class_fraction,
        -record.chain_length,
        record.pdb_id,
        record.resolved_chain,
    )


def select_cath_records(
    candidates: list[CathRecord],
    *,
    quotas: dict[int, int],
    excluded_pdbs: set[str],
    max_per_topology: int,
) -> list[CathRecord]:
    pools: dict[int, list[CathRecord]] = defaultdict(list)
    for record in candidates:
        if record.pdb_id in excluded_pdbs:
            continue
        if record.cath_dominant_class not in quotas:
            continue
        pools[record.cath_dominant_class].append(record)
    for cls in pools:
        pools[cls].sort(key=cath_sort_key)

    selected: list[CathRecord] = []
    used_pdbs: set[str] = set()
    topology_counts: Counter[str] = Counter()

    def topo(record: CathRecord) -> str:
        return f"{record.cath_dominant_class}.{record.cath_dominant_architecture}.{record.cath_dominant_topology}"

    for enforce_topology in [True, False]:
        for cls in sorted(quotas):
            needed = quotas[cls]
            if sum(1 for item in selected if item.cath_dominant_class == cls) >= needed:
                continue
            for record in pools.get(cls, []):
                if sum(1 for item in selected if item.cath_dominant_class == cls) >= needed:
                    break
                if record.pdb_id in used_pdbs:
                    continue
                topology = topo(record)
                if enforce_topology and topology_counts[topology] >= max_per_topology:
                    continue
                selected.append(record)
                used_pdbs.add(record.pdb_id)
                topology_counts[topology] += 1
    return selected


def find_existing_cath_chain(root: Path, pdb_id: str, chain_id: str) -> Path | None:
    tag = safe_chain_filename(chain_id)
    candidates = list(root.glob(f"**/{pdb_id.lower()}_{tag}.cif")) + list(root.glob(f"**/{pdb_id.upper()}_{tag}.cif"))
    return candidates[0] if candidates else None


def build_cath_negative_rows(
    records: list[CathRecord],
    *,
    negative_dir: Path,
    cath_chain_root: Path,
    download_dir: Path,
    timeout: int,
    retries: int,
    backoff: float,
    link_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in records:
        filename = f"{record.pdb_id}_{safe_chain_filename(record.resolved_chain)}.cif"
        target = negative_dir / filename
        source = find_existing_cath_chain(cath_chain_root, record.pdb_id, record.resolved_chain)
        if source is None:
            full = download_cif(record.pdb_id, download_dir / "full_entries", timeout=timeout, retries=retries, backoff=backoff)
            if full is not None:
                generated = download_dir / "chains" / filename
                if extract_chain_cif(full, record.pdb_id, record.resolved_chain, generated):
                    source = generated
        if source is None or not source.exists():
            failures.append({"pdb_id": record.pdb_id, "chain": record.resolved_chain, "reason": "missing_or_extract_failed"})
            continue
        link_or_copy(source, target, mode=link_mode)
        rows.append(
            {
                "record_id": f"cbdneg_cath_{safe_part(record.pdb_id)}_{safe_part(record.resolved_chain)}",
                "filename": filename,
                "pdb_id": record.pdb_id,
                "selected_chain_id": record.resolved_chain,
                "label_chain_id": record.resolved_chain,
                "fallback_chain_id": record.resolved_chain,
                "source_panel": "cath_pisces_hard_negative",
                "source_detail": record.cath_dominant_class_name,
                "source_path": str(source),
                "structure_path": str(target),
                "y_true": 0,
                "cath_dominant_class": record.cath_dominant_class,
                "cath_dominant_topology": f"{record.cath_dominant_class}.{record.cath_dominant_architecture}.{record.cath_dominant_topology}",
                "cath_dominant_superfamily": (
                    f"{record.cath_dominant_class}.{record.cath_dominant_architecture}."
                    f"{record.cath_dominant_topology}.{record.cath_dominant_superfamily}"
                ),
                "chain_length": record.chain_length,
                "resolution": record.resolution,
            }
        )
    return rows, failures


def select_alpha_records(records: list[MpstrucRecord], *, excluded_pdbs: set[str], quota: int) -> list[MpstrucRecord]:
    priority = {"master": 0, "member": 1, "related_master": 2, "related_member": 3}
    unique = [
        record
        for record in deduplicate_mpstruc_records(records)
        if record.pdb_code not in excluded_pdbs
        and "BETA-BARREL" not in record.group_name.upper()
        and "BETA-BARREL" not in record.subgroup_name.upper()
    ]

    def sort_key(record: MpstrucRecord) -> tuple[int, float, str]:
        resolution = parse_float(record.resolution)
        return (priority.get(record.source_type, 99), resolution if resolution is not None else 999.0, record.pdb_code)

    unique.sort(key=sort_key)
    return unique[:quota]


def build_alpha_negative_rows(
    records: list[MpstrucRecord],
    *,
    negative_dir: Path,
    download_dir: Path,
    timeout: int,
    retries: int,
    backoff: float,
    link_mode: str,
    min_residues: int,
    max_success: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in records:
        if max_success is not None and len(rows) >= max_success:
            break
        full = download_cif(record.pdb_code, download_dir / "full_entries", timeout=timeout, retries=retries, backoff=backoff)
        if full is None:
            failures.append({"pdb_id": record.pdb_code, "reason": "download_failed"})
            continue
        picked = largest_protein_chain(full, record.pdb_code, min_residues=min_residues)
        if picked is None:
            failures.append({"pdb_id": record.pdb_code, "reason": "no_large_protein_chain"})
            continue
        chain_id, chain_residues = picked
        filename = f"{record.pdb_code}.cif"
        target = negative_dir / filename
        link_or_copy(full, target, mode=link_mode)
        rows.append(
            {
                "record_id": f"cbdneg_mpalpha_{safe_part(record.pdb_code)}_{safe_part(chain_id)}",
                "filename": filename,
                "pdb_id": record.pdb_code,
                "selected_chain_id": chain_id,
                "label_chain_id": chain_id,
                "fallback_chain_id": chain_id,
                "source_panel": "mpstruc_alpha_helical_membrane",
                "source_detail": record.subgroup_name,
                "source_path": str(full),
                "structure_path": str(target),
                "y_true": 0,
                "mpstruc_name": record.mpstruc_name,
                "mpstruc_source_type": record.source_type,
                "chain_length": chain_residues,
                "resolution": parse_float(record.resolution),
            }
        )
    return rows, failures


def build_benchmark_cohort(positives: list[dict[str, Any]], negatives: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, items in [("positive", positives), ("negative", negatives)]:
        for row in items:
            rows.append(
                {
                    "record_id": row["record_id"],
                    "filename": row["filename"],
                    "pdb_id": row["pdb_id"],
                    "selected_chain_id": row["selected_chain_id"],
                    "label_chain_id": row["label_chain_id"],
                    "fallback_chain_id": row["fallback_chain_id"],
                    "manifest_source": row["source_panel"],
                    "original_split": split,
                    "corrected_split": split,
                    "include_for_metrics": True,
                    "y_true": row["y_true"],
                    "structure_path": row["structure_path"],
                    "structure_exists": Path(str(row["structure_path"])).exists(),
                    "correction_policy": "mpstruc_cath_pisces_public_benchmark",
                    "correction_reason": "",
                    "source_detail": row["source_detail"],
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "source_panel_counts": dict(Counter(str(row["source_panel"]) for row in rows if "source_panel" in row)),
        "source_detail_counts": dict(Counter(str(row["source_detail"]) for row in rows if "source_detail" in row)),
        "unique_pdb": len({str(row["pdb_id"]) for row in rows}),
        "existing_structures": sum(1 for row in rows if Path(str(row["structure_path"])).exists()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--beta-root", type=Path, default=DEFAULT_BETA_ROOT)
    parser.add_argument("--beta-xml", type=Path, default=DEFAULT_BETA_XML)
    parser.add_argument("--beta-classification", type=Path, default=DEFAULT_BETA_CLASSIFICATION)
    parser.add_argument("--cath-candidates", type=Path, default=DEFAULT_CATH_CANDIDATES)
    parser.add_argument("--cath-chain-root", type=Path, default=DEFAULT_CATH_CHAIN_ROOT)
    parser.add_argument("--alpha-url", default=MPSTRUC_ALPHA_URL)
    parser.add_argument("--negative-total", type=int, default=800)
    parser.add_argument("--alpha-quota", type=int, default=267)
    parser.add_argument("--alpha-overselect", type=int, default=64)
    parser.add_argument("--cath-beta-quota", type=int, default=267)
    parser.add_argument("--cath-alpha-quota", type=int, default=133)
    parser.add_argument("--cath-alpha-beta-quota", type=int, default=133)
    parser.add_argument("--max-per-topology", type=int, default=10)
    parser.add_argument("--min-alpha-chain-residues", type=int, default=80)
    parser.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff", type=float, default=1.0)
    parser.add_argument("--force-download-alpha-xml", action="store_true")
    return parser.parse_args()


def main() -> int:
    start = time.perf_counter()
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    source_dir = ensure_dir(out_dir / "source_snapshots")
    positive_dir = ensure_dir(out_dir / "structures/positive")
    negative_dir = ensure_dir(out_dir / "structures/negative")
    downloads_dir = ensure_dir(out_dir / "downloads")
    metadata_dir = ensure_dir(out_dir / "metadata")
    clear_directory(positive_dir)
    clear_directory(negative_dir)

    alpha_xml = download_text(
        args.alpha_url,
        source_dir / "mpstruc_alpha_helical.xml",
        timeout=args.timeout,
        retries=args.retries,
        backoff=args.backoff,
        force=args.force_download_alpha_xml,
    )

    beta_positives = read_beta_positives(
        args.beta_classification.expanduser().resolve(),
        args.beta_root.expanduser().resolve(),
        positive_dir,
        args.link_mode,
    )
    beta_pdbs = {str(row["pdb_id"]) for row in beta_positives}

    alpha_records_all = parse_mpstruc_records(alpha_xml, include_related=True)
    alpha_selected = select_alpha_records(
        alpha_records_all,
        excluded_pdbs=beta_pdbs,
        quota=int(args.alpha_quota) + max(0, int(args.alpha_overselect)),
    )
    alpha_rows_all, alpha_failures = build_alpha_negative_rows(
        alpha_selected,
        negative_dir=negative_dir,
        download_dir=downloads_dir / "mpstruc_alpha",
        timeout=args.timeout,
        retries=args.retries,
        backoff=args.backoff,
        link_mode=args.link_mode,
        min_residues=args.min_alpha_chain_residues,
        max_success=int(args.alpha_quota),
    )
    alpha_rows = alpha_rows_all

    cath_candidates = read_cath_candidates(args.cath_candidates.expanduser().resolve())
    remaining_negative = max(0, int(args.negative_total) - len(alpha_rows))
    cath_beta_quota = min(int(args.cath_beta_quota), remaining_negative)
    cath_remaining = max(0, remaining_negative - cath_beta_quota)
    cath_quotas = {
        1: cath_remaining // 2 + cath_remaining % 2,
        2: cath_beta_quota,
        3: cath_remaining // 2,
    }
    cath_selected = select_cath_records(
        cath_candidates,
        quotas=cath_quotas,
        excluded_pdbs=beta_pdbs | {str(row["pdb_id"]) for row in alpha_rows},
        max_per_topology=int(args.max_per_topology),
    )
    cath_rows, cath_failures = build_cath_negative_rows(
        cath_selected,
        negative_dir=negative_dir,
        cath_chain_root=args.cath_chain_root.expanduser().resolve(),
        download_dir=downloads_dir / "cath_pisces",
        timeout=args.timeout,
        retries=args.retries,
        backoff=args.backoff,
        link_mode=args.link_mode,
    )

    negatives = alpha_rows + cath_rows
    if len(negatives) > args.negative_total:
        negatives = negatives[: args.negative_total]
    cohort = build_benchmark_cohort(beta_positives, negatives)

    duplicate_files = [name for name, count in Counter(str(row["filename"]) for row in cohort).items() if count > 1]
    duplicate_records = [name for name, count in Counter(str(row["record_id"]) for row in cohort).items() if count > 1]
    if duplicate_files or duplicate_records:
        raise RuntimeError(
            f"Duplicate cohort identifiers: files={duplicate_files[:5]} records={duplicate_records[:5]}"
        )
    missing = [row for row in cohort if not Path(str(row["structure_path"])).exists()]
    if missing:
        raise RuntimeError(f"Missing structure paths in cohort; first={missing[:5]}")

    write_csv(out_dir / "positive_manifest_full_mpstruc.csv", beta_positives)
    write_csv(out_dir / "negative_manifest_800.csv", negatives)
    write_csv(out_dir / "benchmark_cohort.csv", cohort)
    write_csv(metadata_dir / "mpstruc_alpha_records.csv", [asdict(record) for record in alpha_records_all])
    write_csv(metadata_dir / "mpstruc_alpha_selected.csv", [asdict(record) for record in alpha_selected])
    write_csv(metadata_dir / "cath_selected.csv", [asdict(record) for record in cath_selected])
    write_csv(metadata_dir / "build_failures.csv", alpha_failures + cath_failures)

    metadata = {
        "schema": "betlas.beta-barrel-detection.full-mpstruc-neg800.v1",
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": round(time.perf_counter() - start, 3),
        "parameters": {
            "negative_total": int(args.negative_total),
            "alpha_quota": int(args.alpha_quota),
            "alpha_overselect": int(args.alpha_overselect),
            "requested_cath_beta_quota": int(args.cath_beta_quota),
            "requested_cath_alpha_quota": int(args.cath_alpha_quota),
            "requested_cath_alpha_beta_quota": int(args.cath_alpha_beta_quota),
            "resolved_cath_quotas": cath_quotas,
            "max_per_topology": int(args.max_per_topology),
            "link_mode": args.link_mode,
        },
        "sources": {
            "mpstruc_beta_xml": file_state(args.beta_xml.expanduser().resolve()),
            "mpstruc_beta_classification": file_state(args.beta_classification.expanduser().resolve()),
            "mpstruc_alpha_url": args.alpha_url,
            "mpstruc_alpha_xml": file_state(alpha_xml),
            "cath_candidates": file_state(args.cath_candidates.expanduser().resolve()),
        },
        "counts": {
            "positive": summarize(beta_positives),
            "negative": summarize(negatives),
            "cohort_rows": len(cohort),
            "metric_rows": sum(1 for row in cohort if bool(row["include_for_metrics"])),
            "alpha_record_rows": len(alpha_records_all),
            "alpha_unique_pdb": len({record.pdb_code for record in alpha_records_all}),
            "alpha_failures": len(alpha_failures),
            "cath_failures": len(cath_failures),
        },
        "directories": {
            "positive_dir": str(positive_dir),
            "negative_dir": str(negative_dir),
        },
        "python": sys.version,
    }
    write_json(out_dir / "metadata.json", metadata)

    print(f"Wrote full-MPStruc beta-barrel detection benchmark to {out_dir}")
    print(f"Positive rows: {len(beta_positives)}")
    print(f"Negative rows: {len(negatives)}")
    print(f"Cohort rows: {len(cohort)}")
    print("Negative source counts:", Counter(row["source_panel"] for row in negatives))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
