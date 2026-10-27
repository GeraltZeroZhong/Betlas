"""Read verified native formats without inventing missing scores or labels."""

from __future__ import annotations

import math
import re
from pathlib import Path


class FormatError(ValueError):
    """An output cannot safely be interpreted as a completed prediction."""


def _number(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise FormatError("Non-finite score")
    return result


def _pair(i: int, j: int, length: int) -> tuple[int, int]:
    if i == j or min(i, j) < 1 or max(i, j) > length:
        raise FormatError(f"Invalid 1-based pair {i}, {j} for length {length}")
    return min(i, j), max(i, j)


def _result(tool: str, sample_id: str, length: int, native_base: int) -> dict:
    if not sample_id or length < 1:
        raise FormatError("A sample identifier and positive sequence length are required")
    return dict(schema_version=1, tool=tool, sample_id=sample_id,
                sequence_length=length, native_index_base=native_base,
                normalized_index_base=1, status="ok", contacts=[],
                strand_pairs=[], strands=[])


def _lines(path: str | Path) -> list[str]:
    return [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]


def read_bcov_strands(path: str | Path, length: int) -> list[dict]:
    """BCov .ss: count, then zero-based index/start/end (inclusive)."""
    lines = _lines(path)
    if not lines:
        raise FormatError("Missing BCov strand count")
    count = int(lines[0])
    if count < 0 or len(lines) != count + 1:
        raise FormatError("BCov strand count mismatch")
    strands = []
    last_end = -1
    for expected, line in enumerate(lines[1:]):
        fields = line.split()
        if len(fields) != 3:
            raise FormatError("BCov strand row must have three integers")
        index, start, end = map(int, fields)
        if index != expected or not last_end < start <= end < length:
            raise FormatError("BCov strands must be ordered, disjoint, zero-based and in range")
        strands.append(dict(strand_id=index + 1, start=start + 1, end=end + 1))
        last_end = end
    return strands


def parse_bcov(topology: str | Path, contacts: str | Path, strands_file: str | Path,
               *, sample_id: str, length: int) -> dict:
    """BCov 1.0 main.c writes hard two-column contacts, despite its README."""
    result = _result("BCov", sample_id, length, 0)
    strands = read_bcov_strands(strands_file, length)
    result["strands"] = strands
    lines = _lines(topology)
    if not lines or int(lines[0]) != len(strands):
        raise FormatError("BCov topology strand count mismatch")
    directions = {"0": "parallel", "1": "antiparallel", "2": "bridge"}
    orient = {}
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != 3 or fields[2] not in directions:
            raise FormatError("Invalid BCov topology row")
        i, j = _pair(int(fields[0]) + 1, int(fields[1]) + 1, len(strands))
        if (i, j) in orient:
            raise FormatError("Duplicate BCov strand pair")
        orient[i, j] = directions[fields[2]]
        result["strand_pairs"].append(dict(strand_i=i, strand_j=j,
            orientation=orient[i, j], score=None, score_kind="hard_prediction"))
    residue_strand = {r: s["strand_id"] for s in strands
                      for r in range(s["start"], s["end"] + 1)}
    lines = _lines(contacts)
    if not lines or int(lines[0]) != length:
        raise FormatError("BCov contact-map sequence length mismatch")
    seen = set()
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != 2:
            raise FormatError("BCov 1.0 contact output has two indices, no confidence column")
        i, j = _pair(int(fields[0]) + 1, int(fields[1]) + 1, length)
        if (i, j) in seen:
            raise FormatError("Duplicate BCov residue pair")
        seen.add((i, j))
        if i not in residue_strand or j not in residue_strand:
            raise FormatError("BCov contact outside supplied strands")
        key = tuple(sorted((residue_strand[i], residue_strand[j])))
        if key not in orient:
            raise FormatError("BCov contact has no corresponding strand-pair prediction")
        result["contacts"].append(dict(residue_i=i, residue_j=j,
            orientation=orient[key], score=None, score_kind="hard_prediction"))
    if not orient and not seen:
        result["status"] = "empty"
    return result


def parse_bbcontacts(path: str | Path, *, sample_id: str, length: int,
                     successful_exit: bool) -> dict:
    """Read .filteredoutput.txt only with independent successful-exit evidence.

    Upstream writes all-NA rows both on errors and when no pattern survives.
    Viterbi scores belong to paths, not calibrated residue probabilities.
    """
    if not successful_exit:
        raise FormatError("bbcontacts requires a successful process exit, including for NA output")
    result = _result("bbcontacts", sample_id, length, 1)
    rows = [x for x in _lines(path) if not x.startswith("#")]
    if not rows:
        raise FormatError("Empty or header-only bbcontacts file is incomplete")
    seen = set()
    for line in rows:
        fields = line.split()
        if len(fields) != 8 or fields[0] != sample_id:
            raise FormatError("bbcontacts must have eight columns and the expected identifier")
        diversity = _number(fields[1])
        if diversity < 0:
            raise FormatError("Negative alignment diversity")
        if fields[2:] == ["NA"] * 6:
            if len(rows) != 1:
                raise FormatError("Mixed empty and nonempty/appended bbcontacts output")
            result.update(status="empty", reported_diversity=diversity)
            return result
        if fields[2] not in ("Parallel", "Antiparallel"):
            raise FormatError("Unknown bbcontacts orientation")
        i, j = _pair(int(fields[6]), int(fields[7]), length)
        if (i, j) in seen:
            raise FormatError("Duplicate bbcontacts residue pair")
        seen.add((i, j))
        path_id = int(fields[4])
        states = {"first", "internal", "last", "bulgei0", "bulgei1", "bulgei2",
                  "bulgej0", "bulgej1", "bulgej2"}
        if path_id < 1 or fields[5] not in states:
            raise FormatError("Unknown bbcontacts contact state or invalid path index")
        result["contacts"].append(dict(residue_i=i, residue_j=j,
            orientation=fields[2].lower(), score=_number(fields[3]),
            score_kind="path_viterbi_score", native_path_id=path_id,
            native_state=fields[5], reported_diversity=diversity))
    return result


def parse_betapro_map(path: str | Path, *, sample_id: str,
                     expected_sequence: str | None = None) -> dict:
    """Read five header lines then a dense matrix over E positions only."""
    lines = Path(path).read_text().splitlines()
    if len(lines) < 5:
        raise FormatError("Truncated BetaPro probability map")
    name, sequence, secondary = [x.strip() for x in lines[:3]]
    if name != sample_id:
        raise FormatError("BetaPro identifier mismatch")
    if not sequence or len(secondary) != len(sequence) or set(secondary) - set("HEC.GIST-"):
        raise FormatError("Invalid BetaPro sequence/secondary structure")
    if expected_sequence is not None and sequence != expected_sequence:
        raise FormatError("BetaPro sequence mismatch")
    positions = [i + 1 for i, ss in enumerate(secondary) if ss == "E"]
    matrix = [list(map(_number, line.split())) for line in lines[5:] if line.strip()]
    if len(matrix) != len(positions) or any(len(r) != len(positions) for r in matrix):
        raise FormatError("BetaPro matrix size must equal the number of E residues")
    result = _result("BetaPro", sample_id, len(sequence), 1)
    result.update(sequence=sequence, secondary_structure=secondary,
                  beta_residue_positions=positions, score_matrix=[])
    for i, row in enumerate(matrix):
        for j, score in enumerate(row):
            if not 0 <= score <= 1:
                raise FormatError("BetaPro probability outside [0,1]")
            if not math.isclose(score, matrix[j][i], abs_tol=1e-5, rel_tol=1e-5):
                raise FormatError("Asymmetric BetaPro probability matrix")
            if i < j:
                result["score_matrix"].append(dict(residue_i=positions[i],
                    residue_j=positions[j], orientation="unspecified", score=score,
                    score_kind="residue_pair_probability"))
    if not positions:
        result["status"] = "empty"
    return result


def parse_betapro_str(path: str | Path, *, sample_id: str,
                     expected_sequence: str | None = None) -> dict:
    """Read BetaPro's human-readable .str with aligned sequence coordinates."""
    text = Path(path).read_text()
    for section in ("Name:", "Sequence:", "The predicted beta-strands:",
                    "The predicted strand pairs and strand alignments:",
                    "The predicted beta-sheets:"):
        if section not in text:
            raise FormatError(f"Incomplete BetaPro .str: missing {section}")
    name = text.split("Name:", 1)[1].strip().splitlines()[0]
    sequence = text.split("Sequence:", 1)[1].strip().splitlines()[0]
    if name != sample_id or (expected_sequence is not None and sequence != expected_sequence):
        raise FormatError("BetaPro identifier or sequence mismatch")
    result = _result("BetaPro", sample_id, len(sequence), 1)
    secondary_match = re.search(r"Secondary Structure[^\n]*\n([^\n]+)", text)
    if secondary_match is None:
        raise FormatError("Missing BetaPro secondary structure")
    secondary = secondary_match.group(1).strip()
    if len(secondary) != len(sequence) or set(secondary) - set("HEC.GIST-"):
        raise FormatError("Invalid BetaPro secondary structure")
    strand_text, pair_text = text.split("The predicted beta-strands:", 1)[1].split(
        "The predicted strand pairs and strand alignments:", 1)
    pair_text = pair_text.split("The predicted beta-sheets:", 1)[0]
    pattern = r"^\s*(\d+):\[(\d+)-(\d+)\]:\[(\d+)-(\d+)\]\s*$"
    covered = 0
    last_end = 0
    for match in re.finditer(pattern, strand_text, re.M):
        sid, start, end, beta_start, beta_end = map(int, match.groups())
        if (sid != len(result["strands"]) + 1 or not last_end < start <= end <= len(sequence)
                or beta_start != covered + 1 or beta_end - beta_start != end - start):
            raise FormatError("Invalid BetaPro strand indexing")
        result["strands"].append(dict(strand_id=sid, start=start, end=end))
        covered, last_end = beta_end, end
    strands = {s["strand_id"]: s for s in result["strands"]}
    annotated = {i + 1 for i, state in enumerate(secondary) if state == "E"}
    mapped = {i for s in strands.values() for i in range(s["start"], s["end"] + 1)}
    if mapped != annotated:
        raise FormatError("BetaPro strand ranges do not cover exactly the E residues")
    directions = {"P": "parallel", "A": "antiparallel", "B": "bridge"}
    pair_pattern = re.compile(r"^\s*(\d+)--(\d+):([PAB]):\[(\d+)-(\d+):(\d+)-(\d+)\]:([^\s]+)\s*$")
    seen = set()
    for line in pair_text.splitlines():
        if not re.match(r"^\s*\d+--", line):
            continue
        match = pair_pattern.fullmatch(line)
        if match is None:
            raise FormatError("Malformed BetaPro alignment row")
        si, sj, d, a, b, c, e, value = match.groups()
        si, sj, a, b, c, e = map(int, (si, sj, a, b, c, e))
        if si not in strands or sj not in strands or si == sj:
            raise FormatError("BetaPro alignment references unknown strands")
        pair = tuple(sorted((si, sj)))
        if pair in seen:
            raise FormatError("Duplicate BetaPro strand pair")
        seen.add(pair)
        if a > b or b-a != abs(e-c):
            raise FormatError("BetaPro aligned spans have unequal lengths")
        if d == "P" and c > e or d == "A" and c < e:
            raise FormatError("BetaPro direction contradicts alignment")
        if not strands[si]["start"] <= a <= b <= strands[si]["end"] or not (
                strands[sj]["start"] <= min(c,e) <= max(c,e) <= strands[sj]["end"]):
            raise FormatError("BetaPro alignment outside native strands")
        score = _number(value)
        result["strand_pairs"].append(dict(strand_i=pair[0], strand_j=pair[1],
            orientation=directions[d], score=score, score_kind="pseudo_binding_energy"))
        for ri, rj in zip(range(a, b+1), range(c, e+(1 if e >= c else -1), 1 if e >= c else -1), strict=True):
            i, j = _pair(ri, rj, len(sequence))
            result["contacts"].append(dict(residue_i=i, residue_j=j,
                orientation=directions[d], score=score, score_kind="strand_pseudo_binding_energy"))
    if not seen:
        result["status"] = "empty"
    return result
