"""Read fixed CATH releases without remapping their classification hierarchy.

The supported formats are CLF 2.0 (12 columns), CNF 2.0 and DOMALL 2.0.
Sequence-cluster identifiers include every parent in the C.A.T.H.S.O.L.I
hierarchy. No download, coordinate lookup or label-dependent selection occurs.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_DOMAIN = re.compile(r"[A-Za-z0-9]{5}[0-9]{2}\Z")
_CHAIN = re.compile(r"[A-Za-z0-9]{5}\Z")
_INTEGER = re.compile(r"-?[0-9]+\Z")
_CODE = re.compile(r"[1-9][0-9]*(?:\.[1-9][0-9]*){0,3}\Z")
_LEVELS = ("class", "architecture", "topology", "homology")
_CLUSTERS = ("s35", "s60", "s95", "s100")
_LIST_COLUMNS = [
    "record_id", "domain_id", "pdb_id", "chain_id", "cath_chain_id",
    *[f"cath_{level}_code" for level in _LEVELS],
    *[f"cath_{level}_cluster_id" for level in _CLUSTERS],
    "cath_s100_sequence_count", "cath_domain_length", "cath_resolution",
    "cath_source_order",
]


def _lines(path: Path) -> Iterator[tuple[int, str]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            text = line.strip()
            if text and not text.startswith("#"):
                yield number, text


def _error(path: Path, number: int, message: str) -> ValueError:
    return ValueError(f"{path}:{number}: {message}")


def _chain_id(value: str) -> str:
    return "" if value == "0" else value


def _check_version(path: Path, version: str) -> None:
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and not line.lstrip().startswith("#"):
                break
            match = re.search(r"CATH VERSION:\s*v?([0-9]+\.[0-9]+\.[0-9]+)", line)
            if match and match.group(1) != version:
                raise ValueError(f"{path}: declared CATH version {match.group(1)} differs from {version}")


def read_domain_list(path: Path, *, class_id: int | None = None) -> pd.DataFrame:
    """Read CLF rows, preserving file order and validating even filtered rows.

    The first row within a complete hierarchy node is its official representative
    under the documented CLF ordering. ``00`` domain suffixes are preserved.
    """
    if class_id is not None and (isinstance(class_id, bool) or class_id < 1):
        raise ValueError("class_id must be a positive integer or None")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    hierarchy_seen: set[tuple[int, ...]] = set()
    for order, (number, text) in enumerate(_lines(path)):
        fields = text.split()
        if len(fields) != 12 or not _DOMAIN.fullmatch(fields[0]):
            raise _error(path, number, "expected a seven-character domain ID and 12 CLF columns")
        domain_id = fields[0]
        try:
            if any(not _INTEGER.fullmatch(value) for value in fields[1:11]):
                raise ValueError
            numbers = [int(value) for value in fields[1:11]]
            resolution = float(fields[11])
            if min(numbers) < 1 or not math.isfinite(resolution) or resolution <= 0:
                raise ValueError
        except ValueError as exc:
            raise _error(path, number, "invalid positive hierarchy, length or resolution value") from exc
        if domain_id in seen:
            raise _error(path, number, f"duplicate domain ID: {domain_id}")
        seen.add(domain_id)
        hierarchy = tuple(numbers[:9])
        if hierarchy in hierarchy_seen:
            raise _error(path, number, "duplicate complete C.A.T.H.S.O.L.I.D identifier")
        hierarchy_seen.add(hierarchy)
        if class_id is not None and numbers[0] != class_id:
            continue
        row: dict[str, object] = {
            "record_id": domain_id,
            "domain_id": domain_id,
            "pdb_id": domain_id[:4].lower(),
            "chain_id": _chain_id(domain_id[4]),
            "cath_chain_id": domain_id[4],
            "cath_s100_sequence_count": numbers[8],
            "cath_domain_length": numbers[9],
            "cath_resolution": resolution,
            "cath_source_order": order,
        }
        for depth, level in enumerate(_LEVELS, 1):
            row[f"cath_{level}_code"] = ".".join(map(str, numbers[:depth]))
        for depth, level in enumerate(_CLUSTERS, 5):
            row[f"cath_{level}_cluster_id"] = ".".join(map(str, numbers[:depth]))
        rows.append(row)
    return pd.DataFrame(rows, columns=_LIST_COLUMNS)


def read_cath_names(path: Path) -> dict[str, str]:
    """Read node descriptions, excluding the representative-domain field."""
    names: dict[str, str] = {}
    for number, text in _lines(path):
        fields = text.split(maxsplit=2)
        if (len(fields) != 3 or not _CODE.fullmatch(fields[0])
                or not (_DOMAIN.fullmatch(fields[1]) or fields[1] == "???????")
                or not fields[2].startswith(":")):
            raise _error(path, number, "expected CNF node, representative domain and colon-prefixed name")
        code = fields[0]
        if code in names:
            raise _error(path, number, f"duplicate CATH name node: {code}")
        names[code] = fields[2][1:].strip()
    return names


@dataclass(frozen=True)
class DomallEntry:
    """Ordered domains and explicitly excluded fragments for one CATH chain."""

    chain_name: str
    domains: tuple[tuple[dict[str, object], ...], ...]
    fragments: tuple[dict[str, object], ...]


def _segment(tokens: list[str], path: Path, number: int) -> dict[str, object]:
    if len(tokens) != 6:
        raise _error(path, number, "truncated DOMALL segment")
    start_chain, start, start_ins, end_chain, end, end_ins = tokens
    if (len(start_chain) != 1 or not start_chain.isascii() or not start_chain.isalnum()
            or end_chain != start_chain):
        raise _error(path, number, "segment endpoints must have the same valid chain character")
    if not _INTEGER.fullmatch(start) or not _INTEGER.fullmatch(end):
        raise _error(path, number, "invalid DOMALL residue number")
    if any(value != "-" and (len(value) != 1 or not value.isascii() or not value.isalnum())
           for value in (start_ins, end_ins)):
        raise _error(path, number, "invalid DOMALL insertion code")
    start_ins = "" if start_ins == "-" else start_ins
    end_ins = "" if end_ins == "-" else end_ins
    return {"chain_id": _chain_id(start_chain), "start": int(start), "start_ins": start_ins,
            "end": int(end), "end_ins": end_ins}


def read_domall(path: Path, *, chain_ids: set[str] | None = None) -> dict[str, DomallEntry]:
    """Read every DOMALL record strictly, retaining only requested chain names.

    Different segments may name different chains; a single segment cannot cross
    chains. Fragment records are parsed and preserved but never included in a domain.
    """
    entries: dict[str, DomallEntry] = {}
    seen: set[str] = set()
    for number, text in _lines(path):
        tokens = text.split()
        if (len(tokens) < 4 or not _CHAIN.fullmatch(tokens[0])
                or not re.fullmatch(r"D[0-9]{2,}", tokens[1])
                or not re.fullmatch(r"F[0-9]{2,}", tokens[2])):
            raise _error(path, number, "invalid DOMALL chain/domain/fragment header")
        chain_name = tokens[0]
        if chain_name in seen:
            raise _error(path, number, f"duplicate DOMALL chain: {chain_name}")
        seen.add(chain_name)
        domain_count, fragment_count = int(tokens[1][1:]), int(tokens[2][1:])
        if domain_count < 1:
            raise _error(path, number, "DOMALL record contains no domain")
        cursor = 3
        domains = []
        for _ in range(domain_count):
            if cursor >= len(tokens) or not tokens[cursor].isdigit() or int(tokens[cursor]) < 1:
                raise _error(path, number, "missing positive DOMALL segment count")
            count = int(tokens[cursor])
            cursor += 1
            segments = []
            for _ in range(count):
                segment = _segment(tokens[cursor:cursor + 6], path, number)
                cursor += 6
                segments.append(segment)
            domains.append(tuple(segments))
        fragments = []
        for _ in range(fragment_count):
            fragment = _segment(tokens[cursor:cursor + 6], path, number)
            cursor += 6
            if cursor >= len(tokens) or not re.fullmatch(r"\([1-9][0-9]*\)", tokens[cursor]):
                raise _error(path, number, "invalid DOMALL fragment residue count")
            fragment["residue_count"] = int(tokens[cursor][1:-1])
            cursor += 1
            fragments.append(fragment)
        if cursor != len(tokens):
            raise _error(path, number, "unconsumed DOMALL fields")
        if chain_ids is None or chain_name in chain_ids:
            entries[chain_name] = DomallEntry(chain_name, tuple(domains), tuple(fragments))
    return entries


def build_release_manifest(
    domain_list: Path, boundaries: Path, names: Path, *, class_id: int = 2, version: str = "4.4.0"
) -> pd.DataFrame:
    """Join fixed-release labels and boundaries, retaining every selected domain.

    ``boundary_status=ok`` means a DOMALL mapping exists, not that coordinates
    have been checked. Unmapped records contain ``segments_json=[]`` and a reason.
    Reversed or overlapping author-number ranges retain the source endpoints and
    have status ``requires_coordinate_order``; never sort their numeric endpoints.
    A CLF ``00`` suffix maps to the only DOMALL domain only when it has no fragments;
    an official ``01`` still maps to domain one, including D01/F00 records.
    """
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("version must identify a fixed release, such as 4.4.0")
    for path in (domain_list, boundaries, names):
        _check_version(path, version)
    frame = read_domain_list(domain_list, class_id=class_id)
    node_names = read_cath_names(names)
    entries = read_domall(boundaries, chain_ids=set(frame["domain_id"].str[:5]))
    domain_ids = set(frame["domain_id"])
    for domain_id in domain_ids:
        if domain_id.endswith("00") and domain_id[:5] + "01" in domain_ids:
            raise ValueError(f"conflicting whole-chain and numbered domain IDs for {domain_id[:5]}")
    mapped, statuses, reasons = [], [], []
    for domain_id in frame["domain_id"]:
        entry = entries.get(domain_id[:5])
        ordinal = int(domain_id[-2:])
        reason = ""
        segments: tuple[dict[str, object], ...] = ()
        if entry is None:
            reason = "missing_domall_chain"
        elif ordinal == 0:
            if len(entry.domains) == 1 and not entry.fragments:
                segments = entry.domains[0]
            else:
                reason = "whole_chain_id_conflicts_with_domall"
        elif ordinal > len(entry.domains):
            reason = "domain_ordinal_absent_from_domall"
        else:
            segments = entry.domains[ordinal - 1]
        status = "unmapped" if reason else "ok"
        if segments:
            flags = []
            for index, segment in enumerate(segments):
                start = (segment["start"], segment["start_ins"])
                end = (segment["end"], segment["end_ins"])
                if start > end:
                    flags.append("nonmonotonic_author_numbering")
                for previous in segments[:index]:
                    p_start = (previous["start"], previous["start_ins"])
                    p_end = (previous["end"], previous["end_ins"])
                    if previous["chain_id"] == segment["chain_id"] and max(start, p_start) <= min(end, p_end):
                        flags.append("overlapping_author_ranges")
            if flags:
                status = "requires_coordinate_order"
                reason = ";".join(sorted(set(flags)))
        mapped.append(json.dumps(segments, separators=(",", ":"), ensure_ascii=True))
        statuses.append(status)
        reasons.append(reason)
    frame["cath_version"] = version
    frame["segments_json"] = mapped
    frame["boundary_status"] = statuses
    frame["boundary_reason"] = reasons
    for level in _LEVELS:
        code_column = f"cath_{level}_code"
        missing = set(frame[code_column]) - node_names.keys()
        if missing:
            raise ValueError(f"CNF is missing {level} nodes: {', '.join(sorted(missing)[:10])}")
        frame[f"cath_{level}_name"] = frame[code_column].map(node_names)
    return frame


def summarize_support(manifest: pd.DataFrame, *, level: str = "architecture") -> pd.DataFrame:
    """Count domains, native hierarchy nodes and sequence clusters before QC."""
    if level not in {"architecture", "topology"}:
        raise ValueError("level must be architecture or topology")
    code = f"cath_{level}_code"
    rows = []
    for label, group in manifest.groupby(code, sort=False):
        row = {code: label, f"cath_{level}_name": group[f"cath_{level}_name"].iloc[0],
               "domain_count": len(group), "pdb_count": group["pdb_id"].nunique(),
               "topology_count": group["cath_topology_code"].nunique(),
               "homology_count": group["cath_homology_code"].nunique(),
               "boundary_ok_count": int(group["boundary_status"].eq("ok").sum())}
        for cluster in _CLUSTERS:
            row[f"{cluster}_count"] = group[f"cath_{cluster}_cluster_id"].nunique()
        rows.append(row)
    rows.sort(key=lambda row: tuple(map(int, str(row[code]).split("."))))
    columns = [code, f"cath_{level}_name", "domain_count", "pdb_count", "topology_count",
               "homology_count", "boundary_ok_count", *[f"{c}_count" for c in _CLUSTERS]]
    return pd.DataFrame(rows, columns=columns)


def select_s35_representatives(
    manifest: pd.DataFrame, official_representatives: Path | None = None, *,
    availability: pd.Series | None = None,
) -> pd.DataFrame:
    """Choose one S35 representative without using geometry or model outputs.

    With no supplied list, use the first CLF member of each S35 node. An optional
    Boolean Series, indexed by every manifest record ID, declares precomputed
    coordinate/boundary availability. Only unavailable official representatives
    may be replaced, by resolution ascending, length descending, then ID. If an
    entire cluster is unavailable, retain its original representative with status
    ``unavailable``. Feature extraction failures must not drive this selection.
    """
    key = "cath_s35_cluster_id"
    if manifest["record_id"].duplicated().any():
        raise ValueError("manifest record IDs must be unique")
    expected = manifest.sort_values("cath_source_order", kind="stable").drop_duplicates(key)
    official = expected.set_index(key)["record_id"].to_dict()
    if official_representatives is not None:
        versions = manifest["cath_version"].drop_duplicates().tolist()
        if len(versions) != 1:
            raise ValueError("representative selection requires one fixed CATH release")
        _check_version(official_representatives, versions[0])
        supplied = read_domain_list(official_representatives)
        supplied = supplied[supplied[key].isin(official)]
        if supplied[key].duplicated().any() or set(supplied[key]) != set(official):
            raise ValueError("official representative list must cover each manifest S35 cluster exactly once")
        proposed = supplied.set_index(key)["record_id"].to_dict()
        members = manifest.set_index("record_id")[key].to_dict()
        if any(members.get(record_id) != cluster for cluster, record_id in proposed.items()):
            raise ValueError("official representatives are absent from their manifest cluster")
        official = proposed
    if availability is None:
        availability = pd.Series(True, index=manifest["record_id"], dtype=bool)
    if (not availability.index.is_unique or set(availability.index) != set(manifest["record_id"])
            or availability.isna().any() or not pd.api.types.is_bool_dtype(availability.dtype)):
        raise ValueError("availability must be a complete Boolean Series indexed by unique record IDs")
    records = []
    for cluster, group in manifest.groupby(key, sort=False):
        chosen_id = official[cluster]
        source = "official" if official_representatives is not None else "clf_first_member"
        status = "available"
        if not bool(availability.loc[chosen_id]):
            eligible = group[group["record_id"].map(availability)]
            if eligible.empty:
                status = "unavailable"
            else:
                selected = eligible.sort_values(
                    ["cath_resolution", "cath_domain_length", "record_id"],
                    ascending=[True, False, True], kind="stable",
                ).iloc[0]
                chosen_id = selected["record_id"]
                source = "availability_fallback"
        row = group.loc[group["record_id"].eq(chosen_id)].iloc[0].to_dict()
        row.update(official_representative_id=official[cluster], representative_source=source,
                   representative_status=status)
        records.append(row)
    columns = [*manifest.columns, "official_representative_id", "representative_source", "representative_status"]
    return pd.DataFrame(records, columns=columns)
