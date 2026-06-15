from __future__ import annotations

import gzip
import random
import re
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ..constants import CATH_URLS, DEFAULT_CATH_DIR, FOLD_LABELS
from ..models import DomainCandidate

_PROP_ARCHES = {"2.105", "2.110", "2.115", "2.120", "2.130", "2.140"}
_PRISM_ARCHES = {"2.90", "2.100"}
_SOLENOID_ARCHES = {"2.150", "2.160"}
_TIM_TOPOLOGIES = {"3.20.20", "3.20.110"}


def ensure_cath_files(cath_dir: Path = DEFAULT_CATH_DIR) -> dict[str, Path]:
    cath_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, url in CATH_URLS.items():
        path = cath_dir / f"cath-b-newest-{key}.gz"
        if key == "all":
            path = cath_dir / "cath-b-newest-all.gz"
        elif key == "names":
            path = cath_dir / "cath-b-newest-names.gz"
        elif key == "s35":
            path = cath_dir / "cath-b-s35-newest.gz"
        if not path.exists() or path.stat().st_size == 0:
            urllib.request.urlretrieve(url, path)
        paths[key] = path
    return paths


def _iter_gzip_lines(path: Path) -> Iterable[str]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def _code_parts(cath_code: str) -> list[str]:
    return [part for part in str(cath_code).split(".") if part != ""]


def cath_architecture_code(cath_code: str) -> str:
    parts = _code_parts(cath_code)
    return ".".join(parts[:2]) if len(parts) >= 2 else cath_code


def cath_topology_code(cath_code: str) -> str:
    parts = _code_parts(cath_code)
    return ".".join(parts[:3]) if len(parts) >= 3 else cath_code


def cath_homology_code(cath_code: str) -> str:
    parts = _code_parts(cath_code)
    return ".".join(parts[:4]) if len(parts) >= 4 else cath_code


def read_cath_names(path: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    for line in _iter_gzip_lines(path):
        code, _, name = line.partition(" ")
        names[code] = name.strip()
    return names


def read_s35_clusters(path: Path) -> dict[str, str]:
    clusters: dict[str, str] = {}
    for line in _iter_gzip_lines(path):
        fields = line.split(maxsplit=3)
        if len(fields) < 3:
            continue
        domain_id, _status, s35_code = fields[:3]
        clusters[domain_id] = s35_code
    return clusters


def fold_label_from_cath(cath_code: str, cath_name: str = "") -> str | None:
    arch = cath_architecture_code(cath_code)
    topology = cath_topology_code(cath_code)
    name = cath_name.lower()

    if topology in _TIM_TOPOLOGIES:
        return "tim_like_beta_alpha_barrel"
    if topology == "2.60.120":
        return "jelly_roll"
    if arch in _PROP_ARCHES:
        return "beta_propeller"
    if arch in _PRISM_ARCHES:
        return "beta_prism"
    if arch in _SOLENOID_ARCHES or re.search(r"\b(beta[- ]?)?solenoid\b", name):
        return "beta_solenoid"
    if arch == "2.40":
        return "beta_barrel"
    if arch in {"2.60", "2.70", "2.102"}:
        return "beta_sandwich"
    return None


def chain_from_chopping(chopping: str, fallback_domain_id: str) -> str:
    matches = re.findall(r":([^,\s]+)", chopping)
    if matches:
        counts = Counter(matches)
        return counts.most_common(1)[0][0]
    return fallback_domain_id[4:5]


def build_cath_label_rows(
    cath_dir: Path = DEFAULT_CATH_DIR,
    *,
    include_putative: bool = False,
) -> list[dict[str, object]]:
    paths = ensure_cath_files(cath_dir)
    names = read_cath_names(paths["names"])
    s35 = read_s35_clusters(paths["s35"])
    rows: list[dict[str, object]] = []
    for line in _iter_gzip_lines(paths["all"]):
        fields = line.split(maxsplit=3)
        if len(fields) < 4:
            continue
        domain_id, status, cath_code, chopping = fields
        if status == "putative" and not include_putative:
            continue

        arch = cath_architecture_code(cath_code)
        topology = cath_topology_code(cath_code)
        homology = cath_homology_code(cath_code)
        cath_name = names.get(homology) or names.get(topology) or names.get(arch, "")
        label = fold_label_from_cath(cath_code, cath_name)
        if label not in FOLD_LABELS:
            continue

        pdb_id = domain_id[:4].lower()
        chain_id = chain_from_chopping(chopping, domain_id)
        rows.append(
            {
                "record_id": domain_id,
                "pdb_id": pdb_id,
                "assembly_id": "1",
                "model_id": 0,
                "chain_id": chain_id,
                "domain_id": domain_id,
                "residue_ranges": chopping,
                "fold_label_final": label,
                "fold_label_candidates": label,
                "evidence_level": "silver",
                "label_source_primary": "CATH-B daily-release newest",
                "label_source_supporting": "CATH architecture/topology name",
                "manual_curator": "",
                "curation_date": "",
                "label_conflict_notes": "",
                "qc_status": "external_source_unreviewed",
                "allowed_for_publication_benchmark": False,
                "discovered_by_betlas": False,
                "cath_status": status,
                "cath_code": cath_code,
                "cath_architecture_code": arch,
                "cath_topology_code": topology,
                "cath_homology_code": homology,
                "cath_s35_cluster_id": s35.get(domain_id, homology),
                "cath_s35_source": "cath_s35" if domain_id in s35 else "homology_fallback",
                "cath_name": cath_name,
            }
        )
    return rows


def _deterministic_score(row: dict[str, object], seed: int) -> tuple[float, str]:
    key = f"{seed}:{row['record_id']}:{row['fold_label_final']}"
    rnd = random.Random(key)
    return rnd.random(), str(row["record_id"])


def balanced_sample_rows(
    rows: list[dict[str, object]],
    *,
    target_per_class: int = 220,
    seed: int = 13,
    max_per_pdb: int = 4,
    initial_max_per_s35: int = 1,
    max_s35_cap: int = 64,
) -> list[dict[str, object]]:
    by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_label[str(row["fold_label_final"])].append(row)

    sampled: list[dict[str, object]] = []
    for label in FOLD_LABELS:
        candidates = sorted(by_label.get(label, []), key=lambda row: _deterministic_score(row, seed))
        selected: list[dict[str, object]] = []
        selected_ids: set[str] = set()
        pdb_counts: Counter[str] = Counter()
        s35_counts: Counter[str] = Counter()
        cap = max(1, initial_max_per_s35)
        max_cap = max(cap, max_s35_cap)
        while len(selected) < target_per_class and cap <= max_cap:
            for row in candidates:
                if len(selected) >= target_per_class:
                    break
                record_id = str(row["record_id"])
                if record_id in selected_ids:
                    continue
                pdb_id = str(row["pdb_id"])
                s35_id = str(row.get("cath_s35_cluster_id", ""))
                if pdb_counts[pdb_id] >= max_per_pdb:
                    continue
                if s35_counts[s35_id] >= cap:
                    continue
                selected.append(row)
                selected_ids.add(record_id)
                pdb_counts[pdb_id] += 1
                s35_counts[s35_id] += 1
            cap *= 2
            if len(selected) >= min(target_per_class, len(candidates)):
                break
        sampled.extend(selected[:target_per_class])
    return sorted(sampled, key=lambda row: (str(row["fold_label_final"]), str(row["record_id"])))


def build_cath_dataset(
    *,
    cath_dir: Path = DEFAULT_CATH_DIR,
    target_per_class: int = 220,
    seed: int = 13,
    include_putative: bool = False,
    all_eligible: bool = False,
    max_per_pdb: int = 4,
    initial_max_per_s35: int = 1,
    max_s35_cap: int = 64,
) -> pd.DataFrame:
    rows = build_cath_label_rows(cath_dir, include_putative=include_putative)
    if all_eligible:
        return pd.DataFrame(
            sorted(rows, key=lambda row: (str(row["fold_label_final"]), str(row["record_id"])))
        )
    sampled = balanced_sample_rows(
        rows,
        target_per_class=target_per_class,
        seed=seed,
        max_per_pdb=max_per_pdb,
        initial_max_per_s35=initial_max_per_s35,
        max_s35_cap=max_s35_cap,
    )
    return pd.DataFrame(sampled)


def domain_from_row(row: dict[str, object]) -> DomainCandidate:
    return DomainCandidate.from_mapping(row)
