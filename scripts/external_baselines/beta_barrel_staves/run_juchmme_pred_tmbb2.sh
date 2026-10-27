#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_juchmme_pred_tmbb2.sh hmm|hnn input.fasta output

Run JUCHMME/PRED-TMBB2 using local HMM or HNN model files.
Required environment: JUCHMME_ROOT (release directory containing bin, models,
tables, and conf). Optional environment: JAVA_BIN (java).
The output captures stdout; output.err captures stderr.
EOF
}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
if [[ $# -ne 3 ]]; then usage >&2; exit 2; fi
MODE="$1"
if [[ "$MODE" != hmm && "$MODE" != hnn ]]; then echo "Error: mode must be 'hmm' or 'hnn'" >&2; exit 2; fi
JUCH="${JUCHMME_ROOT:?Set JUCHMME_ROOT to the local JUCHMME release directory}"
JAVA_BIN="${JAVA_BIN:-java}"
FASTA="$2"
OUT="$3"
if [[ ! -s "$FASTA" ]]; then echo "Error: missing or empty FASTA: $FASTA" >&2; exit 2; fi
if [[ ! -d "$JUCH/bin" ]]; then echo "Error: missing JUCHMME classes: $JUCH/bin" >&2; exit 2; fi
if ! command -v "$JAVA_BIN" >/dev/null 2>&1; then echo "Error: missing Java executable: $JAVA_BIN" >&2; exit 2; fi
mkdir -p "$(dirname "$OUT")"
args=(-cp "$JUCH/bin" hmm.Juchmme -f "$FASTA" -m "$JUCH/models/tmbb2.mdel" -a "$JUCH/tables/A_TMBB2_TRAINED")
case "$MODE" in
  hmm) args+=(-e "$JUCH/tables/E_TMBB2_TRAINED" -c "$JUCH/conf/conf.tmbb");;
  hnn) args+=(-w "$JUCH/tables/W_PREDTMBB2_MYMODEL" -x "$JUCH/tables/BLOSUM62" -c "$JUCH/conf/conf.tmbbHNN");;
esac
"$JAVA_BIN" "${args[@]}" > "$OUT" 2> "$OUT.err"
echo "$OUT"
