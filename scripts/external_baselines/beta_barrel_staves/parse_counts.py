#!/usr/bin/env python3
"""Convert local external-method output to a common strand-count TSV schema."""

from __future__ import annotations

import argparse
import csv
import io
import re
from pathlib import Path
from typing import Any

METHODS = ("betaware", "pred-tmbb2-hmm", "pred-tmbb2-hnn", "proftmb", "tmbed", "polarbearal3")


def count_segments(labels: str, segment_chars: set[str]) -> int:
    """Count contiguous runs of any character in ``segment_chars``."""
    count = 0
    in_segment = False
    for char in labels:
        is_segment = char in segment_chars
        if is_segment and not in_segment:
            count += 1
        in_segment = is_segment
    return count


def prediction(count: int | None, raw: str = "") -> dict[str, Any]:
    return {"strand_count": count, "status": "ok" if count is not None else "failed", "raw": raw}


def parse_tmbed(text: str) -> dict[str, dict[str, Any]]:
    """Read TMbed output format 1: header, sequence, and topology per record."""
    records: dict[str, dict[str, Any]] = {}
    record_id: str | None = None
    body: list[str] = []

    def flush() -> None:
        if record_id is None:
            return
        labels = body[-1] if len(body) >= 2 else ""
        if labels and len(body[0]) != len(labels):
            labels = ""
        records[record_id] = prediction(count_segments(labels, {"B", "b"}) if labels else None, labels)

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            header = line[1:].split()
            if not header:
                raise ValueError("TMbed record has an empty FASTA identifier")
            record_id, body = header[0], []
        else:
            body.append(line)
    flush()
    return records


def parse_juchmme(text: str) -> dict[str, dict[str, Any]]:
    """Count membrane runs in each JUCHMME Viterbi-path (VP) label string."""
    records: dict[str, dict[str, Any]] = {}
    record_id: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("ID: >"):
            header = line.split(">", 1)[1].split()
            if not header:
                raise ValueError("JUCHMME record has an empty identifier")
            record_id = header[0]
            records[record_id] = prediction(None)
        elif record_id is not None and line.startswith("VP:"):
            labels = line.split(":", 1)[1].strip()
            records[record_id] = prediction(count_segments(labels, {"M"}) if labels else None, labels)
    return records


def parse_betaware_topology(text: str) -> str | None:
    """Read BETAWARE's comma-separated strand intervals, including wrapped lines."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not re.match(r"^(Topology|TMB Strands)\s*:", line):
            continue
        topology = [line.split(":", 1)[1].strip()]
        for continuation in lines[index + 1 :]:
            if not re.match(r"^\s+[0-9,\-\s]+$", continuation):
                break
            topology.append(continuation.strip())
        return re.sub(r"\s+", "", "".join(topology))
    return None


def parse_betaware(text: str, record_id: str) -> dict[str, dict[str, Any]]:
    """Parse one BETAWARE result with an explicitly supplied record identifier."""
    topology = parse_betaware_topology(text)
    if topology is None:
        # BETAWARE can omit topology after a negative barrel classification.
        count = 0 if re.search(r"Predicted TMBB\s*:\s*No\b", text) else None
    elif topology == "":
        count = 0
    elif all(re.fullmatch(r"\d+-\d+", item) for item in topology.split(",") if item):
        count = len([item for item in topology.split(",") if item])
    else:
        count = None
    return {record_id: prediction(count, topology or "")}


def parse_proftmb(text: str) -> dict[str, dict[str, Any]]:
    """Read PROFtmb's tabular sequence identifier and predicted strand count."""
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if not {"SeqID", "#_Predicted_Strands"}.issubset(reader.fieldnames or []):
        raise ValueError("PROFtmb output requires SeqID and #_Predicted_Strands columns")
    records = {}
    for row in reader:
        record_id = row["SeqID"].strip().removesuffix(".Q")
        raw = (row["#_Predicted_Strands"] or "").strip()
        count = int(raw) if raw.isdigit() else None
        if not record_id:
            raise ValueError("PROFtmb record has an empty identifier")
        records[record_id] = prediction(count, raw)
    return records


def parse_polarbearal(text: str) -> dict[str, dict[str, Any]]:
    """Read six-column numeric summary lines emitted by PolarBearal3Cli."""
    records = {}
    for line in text.splitlines():
        fields = line.strip().split("\t")
        if len(fields) != 6 or not fields[1].isdigit():
            continue
        try:
            tuple(float(value) for value in fields[2:])
        except ValueError:
            continue
        records[fields[0]] = prediction(int(fields[1]), line.strip())
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-id", help="Required for a single BETAWARE result")
    args = parser.parse_args(argv)
    if args.method == "betaware" and not args.record_id:
        parser.error("--record-id is required for BETAWARE")
    parsers = {
        "tmbed": parse_tmbed,
        "pred-tmbb2-hmm": parse_juchmme,
        "pred-tmbb2-hnn": parse_juchmme,
        "proftmb": parse_proftmb,
        "polarbearal3": parse_polarbearal,
    }
    try:
        text = args.input.read_text(encoding="utf-8", errors="replace")
        records = (parse_betaware(text, args.record_id) if args.method == "betaware"
                   else parsers[args.method](text))
        if not records:
            raise ValueError("No supported prediction records were found")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["record_id", "method", "strand_count", "status", "raw"], delimiter="\t")
            writer.writeheader()
            for record_id, row in records.items():
                writer.writerow({"record_id": record_id, "method": args.method, **row})
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
