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


_INTEGER_RE = re.compile(r"-?\d+")


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = str(value).strip()
    if value in {"", ".", "?"}:
        return None
    if not _INTEGER_RE.fullmatch(value):
        return None
    return int(value)


def _require_numeric_auth_seq(value: str | None, *, context: str) -> int:
    parsed = _safe_int(value)
    if parsed is None:
        raise ValueError(
            "Betlas grammar/slice extraction currently supports numeric author residue IDs only; "
            f"{context} has unsupported value {value!r}"
        )
    return parsed


def _safe_float(value: str | None, default: float = math.nan) -> float:
    try:
        return float(value) if value not in {None, "", ".", "?"} else default
    except (TypeError, ValueError):
        return default


def _normalize_ins_code(value: str | None) -> str:
    value = "" if value is None else str(value).strip()
    return "" if value in {"", ".", "?"} else value


def _display_model_id(value: str | None) -> str:
    value = "" if value is None else str(value).strip()
    if value in {"", ".", "?"}:
        return "0"
    parsed = _safe_int(value)
    if parsed is None:
        return value
    return str(max(0, parsed - 1))


def _model_sort_key(value: str) -> tuple[bool, int, str]:
    parsed = _safe_int(value)
    return (parsed is None, 0 if parsed is None else parsed, value)


def parse_residue_ranges(chopping: str, fallback_chain_id: str = "") -> list[tuple[str, int, int]]:
    ranges: list[tuple[str, int, int]] = []
    for part in str(chopping).split(","):
        part = part.strip()
        if not part:
            continue
        match = re.match(
            r"^\s*(?:(?P<chain_first>[^,:\s]+)\s*:)?"
            r"(?P<start>-?\d+)(?P<start_ins>[A-Za-z]?)\s*-\s*"
            r"(?P<end>-?\d+)(?P<end_ins>[A-Za-z]?)"
            r"(?:\s*:\s*(?P<chain_last>[^,\s]+))?\s*$",
            part,
        )
        if not match:
            raise ValueError(
                f"invalid residue range {part!r}; use '10-180:A', 'A:10-180', or '10-180' with a chain id"
            )
        if match.group("start_ins") or match.group("end_ins"):
            raise ValueError(
                f"unsupported insertion-code residue range {part!r}; Betlas currently supports "
                "numeric author residue ranges only, such as '10-180:A' or 'A:10-180'"
            )
        start = int(match.group("start"))
        end = int(match.group("end"))
        chain_first = match.group("chain_first")
        chain_last = match.group("chain_last")
        if chain_first and chain_last and chain_first != chain_last:
            raise ValueError(
                f"residue range {part!r} has inconsistent chain ids {chain_first!r} and {chain_last!r}"
            )
        chain = chain_last or chain_first or fallback_chain_id
        if not chain:
            raise ValueError(f"residue range {part!r} does not specify a chain id")
        if start <= end:
            ranges.append((chain, start, end))
        else:
            ranges.append((chain, end, start))
    return ranges


def is_mmcif_path(path: str | Path) -> bool:
    """Return whether a path has a supported mmCIF suffix."""

    name = Path(path).name.lower()
    return name.endswith((".cif", ".mmcif", ".cif.gz", ".mmcif.gz"))


def available_auth_chain_ids(path: Path) -> tuple[str, ...]:
    """Return author chain ids present in the atom table of an mmCIF file."""

    mmcif = read_mmcif_dict_for_geometry(path)
    return tuple(sorted({chain for chain in _list_value(mmcif, "_atom_site.auth_asym_id") if chain}))


def inspect_mmcif_chains(path: Path) -> list[dict[str, object]]:
    """Return chain-level mmCIF metadata relevant to Betlas public workflows."""

    mmcif = read_mmcif_dict_for_geometry(path)
    group = _list_value(mmcif, "_atom_site.group_PDB")
    auth_asym = _list_value(mmcif, "_atom_site.auth_asym_id")
    label_asym = _list_value(mmcif, "_atom_site.label_asym_id")
    auth_seq = _list_value(mmcif, "_atom_site.auth_seq_id")
    label_atom = _list_value(mmcif, "_atom_site.label_atom_id")
    auth_atom = _list_value(mmcif, "_atom_site.auth_atom_id")
    comp_id = _list_value(mmcif, "_atom_site.label_comp_id")
    ins_codes = _list_value(mmcif, "_atom_site.pdbx_PDB_ins_code")
    model_nums = _list_value(mmcif, "_atom_site.pdbx_PDB_model_num")

    n = len(auth_asym)
    if n == 0:
        raise ValueError(
            "mmCIF contains no _atom_site.auth_asym_id rows; Betlas cannot inspect chains in an empty "
            "or malformed structure file"
        )
    if not group or len(group) != n:
        group = ["ATOM"] * n
    if not label_asym or len(label_asym) != n:
        label_asym = [""] * n
    if not auth_atom or len(auth_atom) != n:
        auth_atom = label_atom
    if not ins_codes or len(ins_codes) != n:
        ins_codes = ["?"] * n
    if not model_nums or len(model_nums) != n:
        model_nums = [""] * n

    residues_by_chain: dict[str, set[tuple[int, str]]] = defaultdict(set)
    labels_by_chain: dict[str, set[str]] = defaultdict(set)
    model_ids_by_chain: dict[str, set[str]] = defaultdict(set)
    insertion_counts: dict[str, int] = defaultdict(int)
    nonpolymer_counts: dict[str, int] = defaultdict(int)
    protein_like_hetatm_counts: dict[str, int] = defaultdict(int)
    for i in range(n):
        chain = auth_asym[i]
        if not chain:
            continue
        labels_by_chain[chain].add(label_asym[i])
        model_ids_by_chain[chain].add(_display_model_id(model_nums[i]))
        atom_name = (auth_atom[i] or label_atom[i]).strip().upper()
        residue_name = comp_id[i].strip().upper() if i < len(comp_id) else ""
        group_name = group[i].strip().upper()
        protein_like_hetatm = group_name == "HETATM" and residue_name in STANDARD_AMINO_ACIDS
        if group_name not in {"ATOM", "HETATM"} or residue_name not in STANDARD_AMINO_ACIDS:
            nonpolymer_counts[chain] += 1
            continue
        if atom_name != "CA":
            continue
        if protein_like_hetatm:
            protein_like_hetatm_counts[chain] += 1
        seq_id = _safe_int(auth_seq[i])
        if seq_id is None:
            continue
        ins_code = _normalize_ins_code(ins_codes[i])
        residues_by_chain[chain].add((seq_id, ins_code))
        if ins_code:
            insertion_counts[chain] += 1

    sheet_chains = set(_list_value(mmcif, "_struct_sheet_range.beg_auth_asym_id")) | set(
        _list_value(mmcif, "_struct_sheet_range.end_auth_asym_id")
    )
    helix_chains = set(_list_value(mmcif, "_struct_conf.beg_auth_asym_id")) | set(
        _list_value(mmcif, "_struct_conf.end_auth_asym_id")
    )
    sheet_ids = _list_value(mmcif, "_struct_sheet_range.sheet_id")
    range_ids = _list_value(mmcif, "_struct_sheet_range.id")
    beg_chains = _list_value(mmcif, "_struct_sheet_range.beg_auth_asym_id")
    end_chains = _list_value(mmcif, "_struct_sheet_range.end_auth_asym_id")
    beg_seq = _list_value(mmcif, "_struct_sheet_range.beg_auth_seq_id")
    end_seq = _list_value(mmcif, "_struct_sheet_range.end_auth_seq_id")
    beg_ins = _list_value(mmcif, "_struct_sheet_range.pdbx_beg_PDB_ins_code")
    end_ins = _list_value(mmcif, "_struct_sheet_range.pdbx_end_PDB_ins_code")
    if not beg_ins or len(beg_ins) != len(beg_seq):
        beg_ins = ["?"] * len(beg_seq)
    if not end_ins or len(end_ins) != len(end_seq):
        end_ins = ["?"] * len(end_seq)
    usable_sheet_ranges: dict[str, int] = defaultdict(int)
    blocked_sheet_ranges: dict[str, int] = defaultdict(int)
    for _sheet_id, _range_id, beg_chain, end_chain, beg, end, beg_i, end_i in zip(
        sheet_ids, range_ids, beg_chains, end_chains, beg_seq, end_seq, beg_ins, end_ins, strict=False
    ):
        if not beg_chain or beg_chain != end_chain:
            continue
        start = _safe_int(beg)
        stop = _safe_int(end)
        has_insertions = bool(_normalize_ins_code(beg_i) or _normalize_ins_code(end_i))
        if start is None or stop is None or has_insertions:
            blocked_sheet_ranges[beg_chain] += 1
            continue
        if start > stop:
            start, stop = stop, start
        residue_count = sum(
            1 for seq_id, ins_code in residues_by_chain.get(beg_chain, set()) if not ins_code and start <= seq_id <= stop
        )
        if residue_count >= 2:
            usable_sheet_ranges[beg_chain] += 1
        else:
            blocked_sheet_ranges[beg_chain] += 1
    all_chains = sorted(set(auth_asym) | set(residues_by_chain) | sheet_chains | helix_chains)
    rows: list[dict[str, object]] = []
    for chain in all_chains:
        if not chain:
            continue
        residue_count = len(residues_by_chain.get(chain, set()))
        has_sheet = chain in sheet_chains
        has_conf = chain in helix_chains
        hints: list[str] = []
        if residue_count == 0:
            hints.append("no_standard_ca_residues")
        usable_count = int(usable_sheet_ranges.get(chain, 0))
        blocked_count = int(blocked_sheet_ranges.get(chain, 0))
        if has_sheet and insertion_counts.get(chain, 0):
            hints.append("feature_extraction_blocked_insertion_codes")
            hints.append("slice_blocked_insertion_codes")
        elif has_sheet and usable_count > 0:
            hints.append("feature_extraction_supported")
            hints.append("slice_supported")
        elif has_sheet:
            hints.append("sheet_annotations_unusable")
            hints.append("readout_inputs_supported")
        else:
            hints.append("no_sheet_annotations")
            hints.append("readout_inputs_supported")
        if insertion_counts.get(chain, 0):
            hints.append("contains_insertion_codes")
        rows.append(
            {
                "auth_chain_id": chain,
                "label_chain_ids": sorted(label for label in labels_by_chain.get(chain, set()) if label),
                "model_ids": sorted(model_ids_by_chain.get(chain, {"0"}), key=_model_sort_key),
                "standard_ca_residue_count": int(residue_count),
                "sheet_annotation_available": bool(has_sheet),
                "helix_conf_annotation_available": bool(has_conf),
                "insertion_code_ca_count": int(insertion_counts.get(chain, 0)),
                "nonpolymer_atom_rows": int(nonpolymer_counts.get(chain, 0)),
                "protein_like_hetatm_ca_count": int(protein_like_hetatm_counts.get(chain, 0)),
                "usable_sheet_range_count": usable_count,
                "blocked_sheet_range_count": blocked_count,
                "workflow_hints": hints,
            }
        )
    return rows


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
    model_nums = _list_value(mmcif, "_atom_site.pdbx_PDB_model_num")
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
    if not model_nums or len(model_nums) != n:
        model_nums = [""] * n
    allowed_model_nums = {str(int(domain.model_id) + 1)}
    if int(domain.model_id) == 0:
        allowed_model_nums.add("0")

    ranges = parse_residue_ranges(domain.residue_ranges, fallback_chain_id=domain.chain_id)
    chosen: dict[tuple[str, int, str], tuple[float, int, ResidueRecord]] = {}
    for i in range(n):
        model_num = str(model_nums[i]).strip()
        if model_num and model_num not in allowed_model_nums:
            continue
        residue_name = comp_id[i].strip().upper()
        group_name = group[i].strip().upper() if group else "ATOM"
        if group_name not in {"ATOM", "HETATM"} or residue_name not in STANDARD_AMINO_ACIDS:
            continue
        atom_name = (auth_atom[i] or label_atom[i]).strip().upper()
        if atom_name != "CA":
            continue
        chain_id = auth_asym[i]
        if chain_id != domain.chain_id:
            continue
        seq_id = _safe_int(auth_seq[i])
        if seq_id is None:
            seq_id = _require_numeric_auth_seq(
                auth_seq[i],
                context=f"atom_site.auth_seq_id for chain {chain_id}",
            )
        if not _in_ranges(chain_id, seq_id, ranges):
            continue
        alt_id = label_alt[i].strip() if i < len(label_alt) else ""
        alt_rank = 2 if alt_id in {"", ".", "?"} else 1 if alt_id == "A" else 0
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
        if previous is None or (occupancy, alt_rank) > (previous[0], previous[1]):
            chosen[key] = (occupancy, alt_rank, record)

    residues = sorted(
        (record for _occupancy, _alt_rank, record in chosen.values()),
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
    beg_ins = _list_value(mmcif, "_struct_sheet_range.pdbx_beg_PDB_ins_code")
    end_ins = _list_value(mmcif, "_struct_sheet_range.pdbx_end_PDB_ins_code")
    if not beg_ins or len(beg_ins) != len(beg_seq):
        beg_ins = ["?"] * len(beg_seq)
    if not end_ins or len(end_ins) != len(end_seq):
        end_ins = ["?"] * len(end_seq)
    senses = _sheet_sense_by_pair(mmcif)
    ranges = parse_residue_ranges(domain.residue_ranges, fallback_chain_id=domain.chain_id)
    segments: list[SecondaryStructureElement] = []
    for idx, (sheet_id, range_id, beg_chain, end_chain, beg, end, beg_i, end_i) in enumerate(
        zip(sheet_ids, range_ids, beg_chains, end_chains, beg_seq, end_seq, beg_ins, end_ins, strict=False)
    ):
        if beg_chain != domain.chain_id or end_chain != domain.chain_id:
            continue
        start = _safe_int(beg)
        stop = _safe_int(end)
        if start is None or stop is None:
            raise ValueError(
                "Betlas grammar/slice extraction currently supports numeric author residue "
                f"sheet ranges only; mmCIF sheet range {sheet_id}:{range_id} has nonnumeric boundaries"
            )
        range_start = min(start, stop)
        range_stop = max(start, stop)
        if ranges and not any(
            chain == domain.chain_id and not (range_stop < r_start or range_start > r_end)
            for chain, r_start, r_end in ranges
        ):
            continue
        if _normalize_ins_code(beg_i) or _normalize_ins_code(end_i):
            raise ValueError(
                "Betlas grammar/slice extraction currently supports numeric author residue "
                "sheet ranges only; mmCIF sheet range "
                f"{sheet_id}:{range_id} uses insertion-code boundaries"
            )
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
    beg_ins = _list_value(mmcif, "_struct_conf.pdbx_beg_PDB_ins_code")
    end_ins = _list_value(mmcif, "_struct_conf.pdbx_end_PDB_ins_code")
    if not beg_ins or len(beg_ins) != len(beg_seq):
        beg_ins = ["?"] * len(beg_seq)
    if not end_ins or len(end_ins) != len(end_seq):
        end_ins = ["?"] * len(end_seq)
    ranges = parse_residue_ranges(domain.residue_ranges, fallback_chain_id=domain.chain_id)
    helices: list[SecondaryStructureElement] = []
    for conf_id, conf_type, beg_chain, end_chain, beg, end, beg_i, end_i in zip(
        conf_ids, conf_types, beg_chains, end_chains, beg_seq, end_seq, beg_ins, end_ins, strict=False
    ):
        if not conf_type.upper().startswith("HELX"):
            continue
        if beg_chain != domain.chain_id or end_chain != domain.chain_id:
            continue
        start = _safe_int(beg)
        stop = _safe_int(end)
        if start is None or stop is None:
            raise ValueError(
                "Betlas grammar/slice extraction currently supports numeric author residue "
                f"helix ranges only; mmCIF struct_conf {conf_id} has nonnumeric boundaries"
            )
        range_start = min(start, stop)
        range_stop = max(start, stop)
        if ranges and not any(
            chain == domain.chain_id and not (range_stop < r_start or range_start > r_end)
            for chain, r_start, r_end in ranges
        ):
            continue
        if _normalize_ins_code(beg_i) or _normalize_ins_code(end_i):
            raise ValueError(
                "Betlas grammar/slice extraction currently supports numeric author residue "
                f"helix ranges only; mmCIF struct_conf {conf_id} uses insertion-code boundaries"
            )
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
    insertion_residue = next((residue for residue in residues if residue.insertion_code), None)
    if insertion_residue is not None:
        raise ValueError(
            "Betlas grammar/slice extraction currently supports numeric author residue IDs "
            "without insertion codes; selected chain contains insertion-coded residue "
            f"{insertion_residue.chain_id}:{insertion_residue.auth_seq_id}:{insertion_residue.insertion_code}"
        )

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
