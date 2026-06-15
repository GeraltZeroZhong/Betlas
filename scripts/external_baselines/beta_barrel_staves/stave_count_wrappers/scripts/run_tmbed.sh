#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL="$ROOT/tools/TMbed"
PYTHON_BIN="${TMBED_PYTHON:-$ROOT/.conda/tmbed/bin/python}"

FASTA="${1:-$TOOL/examples/sample.fasta}"
EMBEDDINGS="${2:-$TOOL/examples/sample.h5}"
OUT="${3:-$ROOT/runs/tmbed_smoke/sample.pred}"

mkdir -p "$(dirname "$OUT")"
args=(
  -f "$FASTA"
  -p "$OUT"
  --out-format 1
  --no-use-gpu
  --threads "${TMBED_THREADS:-1}"
)

if [[ -n "$EMBEDDINGS" && "$EMBEDDINGS" != "-" ]]; then
  args+=(-e "$EMBEDDINGS")
fi

(
  cd "$TOOL"
  "$PYTHON_BIN" -m tmbed predict "${args[@]}"
)

echo "$OUT"
