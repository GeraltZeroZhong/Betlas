#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_juchmme_pred_tmbb2.sh hmm|hnn input.fasta output

Run the JUCHMME/PRED-TMBB2 baseline wrapper in HMM or HNN mode.
Environment: JAVA_BIN.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 3 ]]; then
  echo "Error: required arguments: hmm|hnn input.fasta output" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JUCH="$ROOT/tools/juchmme_release/juchmme_git"
JAVA_BIN="${JAVA_BIN:-java}"

MODE="$1"
FASTA="$2"
OUT="$3"

if [[ ! -s "$FASTA" ]]; then
  echo "Error: FASTA file does not exist or is empty: $FASTA" >&2
  exit 2
fi
if [[ ! -d "$JUCH/bin" ]]; then
  echo "Error: JUCHMME/PRED-TMBB2 files are missing under $JUCH; run fetch_external_tools.sh first" >&2
  exit 2
fi
if ! command -v "$JAVA_BIN" >/dev/null 2>&1; then
  echo "Error: Java executable is missing: $JAVA_BIN" >&2
  exit 2
fi

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
    echo "Error: mode must be 'hmm' or 'hnn'" >&2
    exit 2
    ;;
esac

echo "$OUT"
