#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_polarbearal.sh [input.pdb] [output_dir]

Build if needed and run the PolarBearal3 command-line shim on one structure.
Environment: DOTNET_BIN, DOTNET_ROOT, POLARBEARAL_REBUILD,
POLARBEARAL_STDOUT, POLARBEARAL_STDERR.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL="$ROOT/tools/PolarBearal3"
MONO_ENV="$ROOT/.conda/polarbearal"
DOTNET_BIN="${DOTNET_BIN:-$MONO_ENV/lib/dotnet/dotnet}"
BUILD_DIR="$ROOT/runs/polarbearal3_build"
DLL="$BUILD_DIR/PolarBearal3Cli.dll"

INPUT="${1:-$ROOT/runs/polarbearal_input/1A0S.pdb}"
OUT_DIR="${2:-$ROOT/runs/polarbearal_smoke}"
STDOUT_PATH="${POLARBEARAL_STDOUT:-$ROOT/runs/polarbearal_smoke.stdout}"
STDERR_PATH="${POLARBEARAL_STDERR:-$ROOT/runs/polarbearal_smoke.stderr}"

if [[ ! -x "$DOTNET_BIN" ]]; then
  DOTNET_BIN="$(command -v dotnet || true)"
fi
if [[ -z "$DOTNET_BIN" ]]; then
  echo ".NET SDK 6 is required. Run scripts/setup_envs.sh first." >&2
  exit 1
fi

mkdir -p "$BUILD_DIR" "$(dirname "$INPUT")" "$OUT_DIR"

if [[ ! -s "$INPUT" ]]; then
  curl -L --fail https://files.rcsb.org/download/1A0S.pdb -o "$INPUT"
fi

# The current upstream PolarBearal3 main branch is missing a break in one menu
# case, which prevents `dotnet build` from compiling. This is a compile-only
# patch and does not change the barrel assignment code paths used below.
if ! perl -0ne 'exit(/case "10":\s*change_to_custom_data\(\);\s*break;\s*default:/ ? 0 : 1)' "$TOOL/Program.cs"; then
  perl -0pi -e 's/(case "10":\s*change_to_custom_data\(\);\s*)(default:)/$1                    break;\n                $2/' "$TOOL/Program.cs"
fi

if [[ "${POLARBEARAL_REBUILD:-0}" == "1" || ! -s "$DLL" ]]; then
  if ! DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1 \
    "$DOTNET_BIN" build "$ROOT/scripts/PolarBearal3Cli.csproj" \
    --configuration Release \
    --output "$BUILD_DIR" \
    > "$BUILD_DIR/build.log" 2>&1; then
    cat "$BUILD_DIR/build.log" >&2
    exit 1
  fi
fi

DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1 \
  DOTNET_ROOT="${DOTNET_ROOT:-$MONO_ENV/lib/dotnet}" \
  "$DOTNET_BIN" "$DLL" "$TOOL" "$INPUT" "$OUT_DIR" \
  > "$STDOUT_PATH" \
  2> "$STDERR_PATH"

echo "$STDOUT_PATH"
