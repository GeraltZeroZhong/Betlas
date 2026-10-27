#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: build_polarbearal.sh [--restore] source_dir build_dir

Build the PolarBearal3 CLI shim using .NET SDK 6 and MathNet.Numerics 5.0.0.
The source files are copied into build_dir/source before a known missing-break
compile fix is applied. The supplied upstream checkout remains unchanged.
Default builds use --no-restore. Pass --restore explicitly to allow NuGet package
restoration, which may access the network. All build files stay in build_dir.
Optional environment: DOTNET_BIN (dotnet), PYTHON_BIN (python3).
EOF
}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
RESTORE=--no-restore
if [[ "${1:-}" == --restore ]]; then RESTORE=; shift; fi
if [[ $# -ne 2 ]]; then usage >&2; exit 2; fi
SOURCE="$(realpath -- "$1")"
BUILD="$(realpath -m -- "$2")"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOTNET_BIN="${DOTNET_BIN:-dotnet}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ! -f "$SOURCE/Program.cs" ]]; then echo "Error: PolarBearal3 source is missing: $SOURCE" >&2; exit 2; fi
if [[ "$SOURCE" == "$BUILD/source" ]]; then echo "Error: source_dir must differ from build_dir/source" >&2; exit 2; fi
if ! command -v "$DOTNET_BIN" >/dev/null 2>&1; then echo "Error: missing .NET SDK: $DOTNET_BIN" >&2; exit 2; fi
mkdir -p "$BUILD/source"
cp -p -- "$SOURCE/"*.cs "$BUILD/source/"
cp -p -- "$HERE/PolarBearal3Cli.csproj" "$HERE/PolarBearal3Cli.cs" "$BUILD/"
"$PYTHON_BIN" - "$BUILD/source/Program.cs" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
fixed = re.sub(
    r'(case "10":\s*change_to_custom_data\(\);\s*)(default:)',
    r'\1break;\n                \2',
    text,
)
if fixed != text:
    path.write_text(fixed)
PY
args=(build "$BUILD/PolarBearal3Cli.csproj" --configuration Release --output "$BUILD/bin")
if [[ -n "$RESTORE" ]]; then args+=("$RESTORE"); fi
DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1 \
  "$DOTNET_BIN" "${args[@]}" > "$BUILD/build.log" 2>&1 || { cat "$BUILD/build.log" >&2; exit 1; }
echo "$BUILD/bin/PolarBearal3Cli.dll"
