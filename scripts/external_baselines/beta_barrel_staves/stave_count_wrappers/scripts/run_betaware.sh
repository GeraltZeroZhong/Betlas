#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_betaware.sh input.fasta input.profile output

Run the BETAWARE stave-count baseline wrapper on explicit local inputs.
Environment: PYTHON_BIN, BETAWARE_THRESHOLD.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 3 ]]; then
  echo "Error: required arguments: input.fasta input.profile output" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL="$ROOT/tools/betaware"
PYTHON_BIN="${PYTHON_BIN:-python}"

FASTA="$1"
PROFILE="$2"
OUT="$3"
THRESHOLD="${BETAWARE_THRESHOLD:-0.8}"

if [[ ! -s "$FASTA" ]]; then
  echo "Error: FASTA file does not exist or is empty: $FASTA" >&2
  exit 2
fi
if [[ ! -s "$PROFILE" ]]; then
  echo "Error: profile file does not exist or is empty: $PROFILE" >&2
  exit 2
fi
if [[ ! -f "$TOOL/betaware.py" ]]; then
  echo "Error: BETAWARE files are missing under $TOOL; run fetch_external_tools.sh first" >&2
  exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: Python executable is missing: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUT")"
BETAWARE_ROOT="$TOOL" "$PYTHON_BIN" "$TOOL/betaware.py" \
  -f "$FASTA" \
  -p "$PROFILE" \
  -s "$THRESHOLD" \
  -t > "$OUT"

echo "$OUT"
