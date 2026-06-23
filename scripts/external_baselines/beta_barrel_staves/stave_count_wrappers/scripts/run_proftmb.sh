#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_proftmb.sh query_file output_prefix

Run the PROFtmb baseline wrapper using the unpacked Debian package files under
tools/proftmb_deb/root.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 2 ]]; then
  echo "Error: required arguments: query_file output_prefix" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG="$ROOT/tools/proftmb_deb/root"
BIN="$PKG/usr/bin/proftmb"
DATA="$PKG/usr/share/proftmb"

QUERY="$1"
OUT_PREFIX="$2"

if [[ ! -s "$QUERY" ]]; then
  echo "Error: query file does not exist or is empty: $QUERY" >&2
  exit 2
fi
if [[ ! -x "$BIN" || ! -d "$DATA" ]]; then
  echo "Error: PROFtmb files are missing under $PKG; run fetch_external_tools.sh first" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUT_PREFIX")"
LD_LIBRARY_PATH="$PKG/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" \
  "$BIN" \
  @"$DATA/options" \
  -d "$DATA" \
  -q "$QUERY" \
  -o "$OUT_PREFIX"

echo "${OUT_PREFIX}_proftmb_tabular.txt"
