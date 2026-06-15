#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JUCH="$ROOT/tools/juchmme_release/juchmme_git"
JAVA_BIN="${JAVA_BIN:-java}"

MODE="${1:-hmm}"
FASTA="${2:-$ROOT/tools/betaware/example/1qj8.fasta}"
OUT="${3:-$ROOT/runs/juchmme_${MODE}_1qj8_smoke.out}"

mkdir -p "$(dirname "$OUT")"

common=(
  -cp "$JUCH/bin" hmm.Juchmme
  -f "$FASTA"
  -m "$JUCH/models/tmbb2.mdel"
  -a "$JUCH/tables/A_TMBB2_TRAINED"
)

case "$MODE" in
  hmm)
    "$JAVA_BIN" "${common[@]}" \
      -e "$JUCH/tables/E_TMBB2_TRAINED" \
      -c "$JUCH/conf/conf.tmbb" \
      > "$OUT" 2> "$OUT.err"
    ;;
  hnn)
    "$JAVA_BIN" "${common[@]}" \
      -w "$JUCH/tables/W_PREDTMBB2_MYMODEL" \
      -x "$JUCH/tables/BLOSUM62" \
      -c "$JUCH/conf/conf.tmbbHNN" \
      > "$OUT" 2> "$OUT.err"
    ;;
  *)
    echo "Usage: $0 [hmm|hnn] [input.fasta] [output]" >&2
    exit 2
    ;;
esac

echo "$OUT"
