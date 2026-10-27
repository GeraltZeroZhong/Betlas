#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_polarbearal.sh input.pdb output_dir

Run the PolarBearal3 command-line shim on the first chain in a local structure.
Required environment: POLARBEARAL_ROOT (upstream source/data directory),
POLARBEARAL_DLL (compiled PolarBearal3Cli.dll).
Optional environment: DOTNET_BIN (dotnet), POLARBEARAL_STDOUT,
POLARBEARAL_STDERR (default: output_dir/polarbearal.stdout and .stderr).
Use build_polarbearal.sh to prepare the shim separately.
EOF
}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
if [[ $# -ne 2 ]]; then usage >&2; exit 2; fi
TOOL="${POLARBEARAL_ROOT:?Set POLARBEARAL_ROOT to the local PolarBearal3 directory}"
DLL="${POLARBEARAL_DLL:?Set POLARBEARAL_DLL to the compiled shim}"
DOTNET_BIN="${DOTNET_BIN:-dotnet}"
if [[ ! -s "$1" ]]; then echo "Error: missing or empty input PDB: $1" >&2; exit 2; fi
if [[ ! -f "$TOOL/Program.cs" || ! -s "$DLL" ]]; then echo "Error: PolarBearal3 source or compiled shim is missing" >&2; exit 2; fi
if ! command -v "$DOTNET_BIN" >/dev/null 2>&1; then echo "Error: missing .NET executable: $DOTNET_BIN" >&2; exit 2; fi
INPUT="$(realpath -- "$1")"
OUT_DIR="$(realpath -m -- "$2")"
STDOUT_PATH="${POLARBEARAL_STDOUT:-$OUT_DIR/polarbearal.stdout}"
STDERR_PATH="${POLARBEARAL_STDERR:-$OUT_DIR/polarbearal.stderr}"
mkdir -p "$OUT_DIR" "$(dirname "$STDOUT_PATH")" "$(dirname "$STDERR_PATH")"
DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1 \
  "$DOTNET_BIN" "$DLL" "$TOOL" "$INPUT" "$OUT_DIR" > "$STDOUT_PATH" 2> "$STDERR_PATH"
echo "$STDOUT_PATH"
