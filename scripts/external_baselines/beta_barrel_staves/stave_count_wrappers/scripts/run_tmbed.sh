#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_tmbed.sh input.fasta embeddings.h5|- output

Run the TMbed baseline wrapper. Pass '-' as the embedding argument to skip the
embedding-cache option.
Environment: TMBED_PYTHON, TMBED_THREADS.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 3 ]]; then
  echo "Error: required arguments: input.fasta embeddings.h5|- output" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL="$ROOT/tools/TMbed"
PYTHON_BIN="${TMBED_PYTHON:-$ROOT/.conda/tmbed/bin/python}"

FASTA="$1"
EMBEDDINGS="$2"
OUT="$3"

if [[ ! -s "$FASTA" ]]; then
  echo "Error: FASTA file does not exist or is empty: $FASTA" >&2
  exit 2
fi
if [[ "$EMBEDDINGS" != "-" && ! -s "$EMBEDDINGS" ]]; then
  echo "Error: embeddings file does not exist or is empty: $EMBEDDINGS" >&2
  exit 2
fi
if [[ ! -d "$TOOL" ]]; then
  echo "Error: TMbed source tree is missing under $TOOL; run fetch_external_tools.sh first" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python || true)"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Error: Python executable for TMbed is missing; set TMBED_PYTHON or run setup_envs.sh" >&2
  exit 2
fi

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
