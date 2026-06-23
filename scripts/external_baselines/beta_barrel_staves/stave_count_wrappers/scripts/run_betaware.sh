#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_betaware.sh [input.fasta] [input.profile] [output]

Run the BETAWARE stave-count baseline wrapper. Defaults use the upstream
example inputs and write to runs/betaware_1qj8_smoke.out.
Environment: PYTHON_BIN, BETAWARE_THRESHOLD.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

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
