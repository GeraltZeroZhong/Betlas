from __future__ import annotations

import gzip
import math
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from ..constants import STANDARD_AMINO_ACIDS
from ..models import (
    DomainCandidate,
    ResidueRecord,
    SecondaryStructureElement,
    SheetPatch,
    StructureGeometry,
)


@contextmanager
def _plain_mmcif_path(path: Path) -> Iterator[Path]:
    if path.suffix.lower() != ".gz":
        yield path
        return
    fd, tmp = tempfile.mkstemp(suffix=".cif")
    os.close(fd)
    try:
        with gzip.open(path, "rb") as source, open(tmp, "wb") as target:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
        yield Path(tmp)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def read_mmcif_dict(path: Path) -> dict[str, object]:
    with _plain_mmcif_path(path) as plain:
        return MMCIF2Dict(str(plain))


@lru_cache(maxsize=32)
def _read_mmcif_dict_cached(path_string: str, size: int, mtime_ns: int) -> dict[str, object]:
    _ = (size, mtime_ns)
    return read_mmcif_dict(Path(path_string))


def read_mmcif_dict_for_geometry(path: Path, *, use_cache: bool = True) -> dict[str, object]:
    if not use_cache:
        return read_mmcif_dict(path)
    stat = path.stat()
    return _read_mmcif_dict_cached(str(path), int(stat.st_size), int(stat.st_mtime_ns))


def _list_value(mmcif: dict[str, object], key: str) -> list[str]:
    value = mmcif.get(key)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = str(value).strip()
    if value in {"", ".", "?"}:
        return None
    match = re.search(r"-?\d+", value)
    if not match:
        return None
    return int(match.group(0))


def _safe_float(value: str | None, default: float = math.nan) -> float:
    try:
        return float(value) if value not in {None, "", ".", "?"} else default
    except (TypeError, ValueError):
        return default


def _normalize_ins_code(value: str | None) -> str:
    value = "" if value is None else str(value).strip()
    return "" if value in {"", ".", "?"} else value


def parse_residue_ranges(chopping: str, fallback_chain_id: str = "") -> list[tuple[str, int, int]]:
    ranges: list[tuple[str, int, int]] = []
    for part in str(chopping).split(","):
        part = part.strip()
        if not part:
            continue
        match = re.match(r"^\s*(-?\d+)[A-Za-z]?\s*-\s*(-?\d+)[A-Za-z]?\s*:([^,\s]+)\s*$", part)
        if not match:
            continue
        start = int(match.group(1))
        end = int(match.group(2))
        chain = match.group(3) or fallback_chain_id
        if start <= end:
            ranges.append((chain, start, end))
        else:
            ranges.append((chain, end, start))
    return ranges


def _in_ranges(chain_id: str, auth_seq_id: int, ranges: list[tuple[str, int, int]]) -> bool:
    if not ranges:
        return True
    return any(chain == chain_id and start <= auth_seq_id <= end for chain, start, end in ranges)


def _selected_ca_records(
    mmcif: dict[str, object],
    domain: DomainCandidate,
) -> tuple[list[ResidueRecord], dict[tuple[str, int, str], int]]:
    group = _list_value(mmcif, "_atom_site.group_PDB")
    auth_asym = _list_value(mmcif, "_atom_site.auth_asym_id")
    auth_seq = _list_value(mmcif, "_atom_site.auth_seq_id")
    label_seq = _list_value(mmcif, "_atom_site.label_seq_id")
    label_atom = _list_value(mmcif, "_atom_site.label_atom_id")
    auth_atom = _list_value(mmcif, "_atom_site.auth_atom_id")
    label_alt = _list_value(mmcif, "_atom_site.label_alt_id")
    comp_id = _list_value(mmcif, "_atom_site.label_comp_id")
    ins_codes = _list_value(mmcif, "_atom_site.pdbx_PDB_ins_code")
    occupancies = _list_value(mmcif, "_atom_site.occupancy")
    xs = _list_value(mmcif, "_atom_site.Cartn_x")
    ys = _list_value(mmcif, "_atom_site.Cartn_y")
    zs = _list_value(mmcif, "_atom_site.Cartn_z")

    n = len(auth_asym)
    if not ins_codes or len(ins_codes) != n:
        ins_codes = ["?"] * n
    if not occupancies or len(occupancies) != n:
        occupancies = ["1.0"] * n
    if not auth_atom or len(auth_atom) != n:
        auth_atom = label_atom

    ranges = parse_residue_ranges(domain.residue_ranges, fallback_chain_id=domain.chain_id)
    chosen: dict[tuple[str, int, str], tuple[float, ResidueRecord]] = {}
    for i in range(n):
        if group and group[i] != "ATOM":
            continue
        atom_name = (auth_atom[i] or label_atom[i]).strip().upper()
        if atom_name != "CA":
            continue
        residue_name = comp_id[i].strip().upper()
        if residue_name not in STANDARD_AMINO_ACIDS:
            continue
        seq_id = _safe_int(auth_seq[i])
        if seq_id is None:
            continue
        chain_id = auth_asym[i]
        if chain_id != domain.chain_id:
            continue
        if not _in_ranges(chain_id, seq_id, ranges):
            continue
        alt_id = label_alt[i].strip() if i < len(label_alt) else ""
        if alt_id not in {"", ".", "?", "A"}:
            continue
        label_seq_id = _safe_int(label_seq[i]) if i < len(label_seq) else None
        ins_code = _normalize_ins_code(ins_codes[i])
        x, y, z = _safe_float(xs[i]), _safe_float(ys[i]), _safe_float(zs[i])
        if not np.isfinite([x, y, z]).all():
            continue
        occupancy = _safe_float(occupancies[i], default=1.0)
        key = (chain_id, seq_id, ins_code)
        record = ResidueRecord(
            pdb_id=domain.pdb_id,
            chain_id=chain_id,
            auth_seq_id=seq_id,
            label_seq_id=label_seq_id,
            insertion_code=ins_code,
            residue_name=residue_name,
            coord_ca=(float(x), float(y), float(z)),
        )
        previous = chosen.get(key)
        if previous is None or occupancy > previous[0]:
            chosen[key] = (occupancy, record)

    residues = sorted(
        (record for _occupancy, record in chosen.values()),
        key=lambda residue: (residue.auth_seq_id, residue.insertion_code),
    )
    index_by_key = {
        (residue.chain_id, residue.auth_seq_id, residue.insertion_code): index
        for index, residue in enumerate(residues)
    }
    return residues, index_by_key


def _segment_indices_for_range(
    residues: list[ResidueRecord],
    chain_id: str,
    start: int,
    end: int,
) -> tuple[int, ...]:
    if start > end:
        start, end = end, start
    return tuple(
        index
        for index, residue in enumerate(residues)
        if residue.chain_id == chain_id and start <= residue.auth_seq_id <= end
    )


def _sheet_sense_by_pair(mmcif: dict[str, object]) -> dict[tuple[str, str], str]:
    sheet_ids = _list_value(mmcif, "_struct_sheet_order.sheet_id")
    range_1 = _list_value(mmcif, "_struct_sheet_order.range_id_1")
    range_2 = _list_value(mmcif, "_struct_sheet_order.range_id_2")
    senses = _list_value(mmcif, "_struct_sheet_order.sense")
    out: dict[tuple[str, str], str] = {}
    for sheet_id, r1, r2, sense in zip(sheet_ids, range_1, range_2, senses, strict=False):
        out[(sheet_id, r2)] = sense
        out[(sheet_id, r1)] = out.get((sheet_id, r1), "")
    return out


def _parse_beta_segments(
    mmcif: dict[str, object],
    residues: list[ResidueRecord],
    domain: DomainCandidate,
) -> list[SecondaryStructureElement]:
    sheet_ids = _list_value(mmcif, "_struct_sheet_range.sheet_id")
    range_ids = _list_value(mmcif, "_struct_sheet_range.id")
    beg_chains = _list_value(mmcif, "_struct_sheet_range.beg_auth_asym_id")
    end_chains = _list_value(mmcif, "_struct_sheet_range.end_auth_asym_id")
    beg_seq = _list_value(mmcif, "_struct_sheet_range.beg_auth_seq_id")
    end_seq = _list_value(mmcif, "_struct_sheet_range.end_auth_seq_id")
    senses = _sheet_sense_by_pair(mmcif)
    ranges = parse_residue_ranges(domain.residue_ranges, fallback_chain_id=domain.chain_id)
    segments: list[SecondaryStructureElement] = []
    for idx, (sheet_id, range_id, beg_chain, end_chain, beg, end) in enumerate(
        zip(sheet_ids, range_ids, beg_chains, end_chains, beg_seq, end_seq, strict=False)
    ):
        if beg_chain != domain.chain_id or end_chain != domain.chain_id:
            continue
        start = _safe_int(beg)
        stop = _safe_int(end)
        if start is None or stop is None:
            continue
        if ranges and not any(
            chain == domain.chain_id and not (stop < r_start or start > r_end)
            for chain, r_start, r_end in ranges
        ):
            continue
        indices = _segment_indices_for_range(residues, domain.chain_id, start, stop)
        if len(indices) < 2:
            continue
        start_residue = residues[indices[0]]
        end_residue = residues[indices[-1]]
        segments.append(
            SecondaryStructureElement(
                element_id=f"B{len(segments) + 1:03d}",
                element_type="strand",
                chain_id=domain.chain_id,
                start_auth_seq_id=start_residue.auth_seq_id,
                end_auth_seq_id=end_residue.auth_seq_id,
                residue_indices=indices,
                sheet_id=sheet_id or f"sheet_{idx + 1}",
                sheet_range_id=range_id,
                sense_to_previous=senses.get((sheet_id, range_id), ""),
            )
        )
    return segments


def _parse_helices(
    mmcif: dict[str, object],
    residues: list[ResidueRecord],
    domain: DomainCandidate,
) -> list[SecondaryStructureElement]:
    conf_types = _list_value(mmcif, "_struct_conf.conf_type_id")
    conf_ids = _list_value(mmcif, "_struct_conf.id")
    beg_chains = _list_value(mmcif, "_struct_conf.beg_auth_asym_id")
    end_chains = _list_value(mmcif, "_struct_conf.end_auth_asym_id")
    beg_seq = _list_value(mmcif, "_struct_conf.beg_auth_seq_id")
    end_seq = _list_value(mmcif, "_struct_conf.end_auth_seq_id")
    ranges = parse_residue_ranges(domain.residue_ranges, fallback_chain_id=domain.chain_id)
    helices: list[SecondaryStructureElement] = []
    for conf_id, conf_type, beg_chain, end_chain, beg, end in zip(
        conf_ids, conf_types, beg_chains, end_chains, beg_seq, end_seq, strict=False
    ):
        if not conf_type.upper().startswith("HELX"):
            continue
        if beg_chain != domain.chain_id or end_chain != domain.chain_id:
            continue
        start = _safe_int(beg)
        stop = _safe_int(end)
        if start is None or stop is None:
            continue
        if ranges and not any(
            chain == domain.chain_id and not (stop < r_start or start > r_end)
            for chain, r_start, r_end in ranges
        ):
            continue
        indices = _segment_indices_for_range(residues, domain.chain_id, start, stop)
        if len(indices) < 3:
            continue
        helices.append(
            SecondaryStructureElement(
                element_id=f"H{len(helices) + 1:03d}",
                element_type="helix",
                chain_id=domain.chain_id,
                start_auth_seq_id=residues[indices[0]].auth_seq_id,
                end_auth_seq_id=residues[indices[-1]].auth_seq_id,
                residue_indices=indices,
                sheet_id="",
                sheet_range_id=conf_id,
            )
        )
    return helices


def build_structure_geometry(
    domain: DomainCandidate,
    mmcif_path: Path,
    *,
    use_cache: bool = True,
) -> StructureGeometry:
    mmcif = read_mmcif_dict_for_geometry(mmcif_path, use_cache=use_cache)
    residues, _index_by_key = _selected_ca_records(mmcif, domain)
    warnings: list[str] = []
    if not residues:
        warnings.append("no_selected_ca_residues")

    beta_segments = _parse_beta_segments(mmcif, residues, domain)
    helices = _parse_helices(mmcif, residues, domain)
    if not beta_segments:
        warnings.append("no_struct_sheet_range_segments")

    by_sheet: dict[str, list[str]] = defaultdict(list)
    for segment in beta_segments:
        by_sheet[segment.sheet_id or "unknown"].append(segment.element_id)
    sheet_patches = [
        SheetPatch(sheet_id=sheet_id, strand_ids=tuple(strand_ids))
        for sheet_id, strand_ids in sorted(by_sheet.items())
    ]

    return StructureGeometry(
        domain=domain,
        residues=tuple(residues),
        beta_segments=tuple(beta_segments),
        helices=tuple(helices),
        sheet_patches=tuple(sheet_patches),
        warnings=tuple(warnings),
    )
