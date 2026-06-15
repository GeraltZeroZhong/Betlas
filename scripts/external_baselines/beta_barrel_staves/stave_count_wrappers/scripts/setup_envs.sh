#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAMBA="${MAMBA:-mamba}"

if ! command -v "$MAMBA" >/dev/null 2>&1; then
  echo "mamba is required. Set MAMBA=/path/to/mamba if it is not on PATH." >&2
  exit 1
fi

if [[ ! -x "$ROOT/.conda/polarbearal/bin/mono" ]]; then
  "$MAMBA" create -y -p "$ROOT/.conda/polarbearal" -c conda-forge mono dotnet-sdk=6
elif [[ ! -x "$ROOT/.conda/polarbearal/lib/dotnet/dotnet" ]]; then
  "$MAMBA" install -y -p "$ROOT/.conda/polarbearal" -c conda-forge dotnet-sdk=6
fi

if [[ ! -x "$ROOT/.conda/tmbed/bin/python" ]]; then
  "$MAMBA" create -y -p "$ROOT/.conda/tmbed" -c conda-forge \
    python=3.11 h5py numpy sentencepiece pytorch-cpu transformers=4.51.3 typer tqdm
fi

echo "External baseline environments are ready."
