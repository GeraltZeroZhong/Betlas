#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_tmbed.sh input.fasta embeddings.h5|- output

Run TMbed in CPU mode with local model files. Pass '-' for embeddings to let
TMbed generate them using its locally available language model.
Required environment: TMBED_ROOT (TMbed source directory).
Optional environment: TMBED_PYTHON (python), TMBED_THREADS (1).
Hugging Face offline mode is enabled; arrange local weights before execution.
EOF
}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
if [[ $# -ne 3 ]]; then usage >&2; exit 2; fi
TOOL="${TMBED_ROOT:?Set TMBED_ROOT to the local TMbed source directory}"
PYTHON_BIN="${TMBED_PYTHON:-python}"
if [[ ! -s "$1" ]]; then echo "Error: missing or empty FASTA: $1" >&2; exit 2; fi
if [[ "$2" != - && ! -s "$2" ]]; then echo "Error: missing or empty embeddings: $2" >&2; exit 2; fi
if [[ ! -d "$TOOL" ]]; then echo "Error: missing TMbed source directory: $TOOL" >&2; exit 2; fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then echo "Error: missing Python executable: $PYTHON_BIN" >&2; exit 2; fi
PYTHON_BIN="$(command -v "$PYTHON_BIN")"
if [[ "$PYTHON_BIN" == */* ]]; then PYTHON_BIN="$(realpath -m -s -- "$PYTHON_BIN")"; fi
FASTA="$(realpath -- "$1")"
OUT="$(realpath -m -- "$3")"
mkdir -p "$(dirname "$OUT")"
args=(-f "$FASTA" -p "$OUT" --out-format 1 --no-use-gpu --threads "${TMBED_THREADS:-1}")
if [[ "$2" != - ]]; then args+=(-e "$(realpath -- "$2")"); fi
(
  cd "$TOOL"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    "$PYTHON_BIN" -m tmbed predict "${args[@]}"
)
echo "$OUT"
