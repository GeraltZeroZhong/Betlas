#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG="$ROOT/tools/proftmb_deb/root"
BIN="$PKG/usr/bin/proftmb"
DATA="$PKG/usr/share/proftmb"

QUERY="${1:-$PKG/usr/share/doc/proftmb/examples/example.Q}"
OUT_PREFIX="${2:-$ROOT/runs/proftmb_smoke/result}"

mkdir -p "$(dirname "$OUT_PREFIX")"
LD_LIBRARY_PATH="$PKG/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" \
  "$BIN" \
  @"$DATA/options" \
  -d "$DATA" \
  -q "$QUERY" \
  -o "$OUT_PREFIX"

echo "${OUT_PREFIX}_proftmb_tabular.txt"
