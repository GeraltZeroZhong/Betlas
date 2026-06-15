#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL="$ROOT/tools/betaware"
PYTHON_BIN="${PYTHON_BIN:-python}"

FASTA="${1:-$TOOL/example/1qj8.fasta}"
PROFILE="${2:-$TOOL/example/1qj8.prof}"
OUT="${3:-$ROOT/runs/betaware_1qj8_smoke.out}"
THRESHOLD="${BETAWARE_THRESHOLD:-0.8}"

mkdir -p "$(dirname "$OUT")"
BETAWARE_ROOT="$TOOL" "$PYTHON_BIN" "$TOOL/betaware.py" \
  -f "$FASTA" \
  -p "$PROFILE" \
  -s "$THRESHOLD" \
  -t > "$OUT"

echo "$OUT"
