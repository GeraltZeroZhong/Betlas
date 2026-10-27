#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_proftmb.sh query_file_or_directory output_prefix

Run PROFtmb with an explicit profile query or profile directory.
Required environment: PROFTMB_ROOT (unpacked installation prefix containing
usr/bin/proftmb and usr/share/proftmb). Optional overrides: PROFTMB_BIN,
PROFTMB_DATA, PROFTMB_LIBRARY_DIR.
EOF
}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
if [[ $# -ne 2 ]]; then usage >&2; exit 2; fi
PKG="${PROFTMB_ROOT:?Set PROFTMB_ROOT to the local PROFtmb installation prefix}"
BIN="${PROFTMB_BIN:-$PKG/usr/bin/proftmb}"
DATA="${PROFTMB_DATA:-$PKG/usr/share/proftmb}"
LIBRARY_DIR="${PROFTMB_LIBRARY_DIR:-$PKG/usr/lib/x86_64-linux-gnu}"
QUERY="$1"
OUT_PREFIX="$2"
if [[ ! -s "$QUERY" && ! -d "$QUERY" ]]; then echo "Error: query file or directory is missing: $QUERY" >&2; exit 2; fi
if [[ ! -x "$BIN" || ! -d "$DATA" ]]; then echo "Error: PROFtmb executable or data directory is missing" >&2; exit 2; fi
mkdir -p "$(dirname "$OUT_PREFIX")"
LD_LIBRARY_PATH="$LIBRARY_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$BIN" @"$DATA/options" -d "$DATA" -q "$QUERY" -o "$OUT_PREFIX"
echo "${OUT_PREFIX}_proftmb_tabular.txt"
