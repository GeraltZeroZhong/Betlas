#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_betaware.sh input.fasta input.profile output

Run BETAWARE on explicit local inputs.
Required environment: BETAWARE_ROOT (directory containing betaware.py).
Optional environment: PYTHON_BIN (python), BETAWARE_THRESHOLD (0.8),
BETAWARE_AA_ORDER (profile column order; upstream default when omitted).
EOF
}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
if [[ $# -ne 3 ]]; then usage >&2; exit 2; fi
TOOL="${BETAWARE_ROOT:?Set BETAWARE_ROOT to the local BETAWARE source directory}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FASTA="$1"
PROFILE="$2"
OUT="$3"
for INPUT in "$FASTA" "$PROFILE"; do
  if [[ ! -s "$INPUT" ]]; then echo "Error: missing or empty input: $INPUT" >&2; exit 2; fi
done
if [[ ! -f "$TOOL/betaware.py" ]]; then echo "Error: missing BETAWARE source: $TOOL/betaware.py" >&2; exit 2; fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then echo "Error: missing Python executable: $PYTHON_BIN" >&2; exit 2; fi
mkdir -p "$(dirname "$OUT")"
args=(-f "$FASTA" -p "$PROFILE" -s "${BETAWARE_THRESHOLD:-0.8}" -t)
if [[ -n "${BETAWARE_AA_ORDER:-}" ]]; then args+=(-a "$BETAWARE_AA_ORDER"); fi
BETAWARE_ROOT="$TOOL" "$PYTHON_BIN" "$TOOL/betaware.py" "${args[@]}" > "$OUT"
echo "$OUT"
