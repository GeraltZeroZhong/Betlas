#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATUS_DIR="${SMOKE_STATUS_DIR:-$ROOT/runs/smoke_status}"
STATUS_PYTHON="${PYTHON_BIN:-python}"
STATUS_WRITER="$ROOT/scripts/write_external_status.py"
mkdir -p "$STATUS_DIR"

manifests=()
overall_status=0

argv_json() {
  "$STATUS_PYTHON" -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@"
}

run_method() {
  local label="$1"
  local input_spec="$2"
  local output_spec="$3"
  shift 3
  local cmd=("$@")
  local stdout_file stderr_file started ended rc status message manifest
  stdout_file="$(mktemp)"
  stderr_file="$(mktemp)"
  started="$(date +%s)"

  echo "$label"
  set +e
  "${cmd[@]}" >"$stdout_file" 2>"$stderr_file"
  rc=$?
  set -e
  ended="$(date +%s)"

  cat "$stdout_file"
  cat "$stderr_file" >&2
  if [[ "$rc" -eq 0 ]]; then
    status="ok"
  else
    status="error"
    overall_status=1
  fi
  message="$(tr '\r\n\t' '   ' <"$stderr_file" | cut -c 1-500)"
  manifest="$STATUS_DIR/${label}.json"

  status_args=(
    "$STATUS_PYTHON" "$STATUS_WRITER" method
    --out "$manifest"
    --method "$label"
    --status "$status"
    --return-code "$rc"
    --runtime-seconds "$((ended - started))"
    --command-json "$(argv_json "${cmd[@]}")"
    --message "$message"
  )
  if [[ -n "$input_spec" ]]; then
    status_args+=(--input "$input_spec")
  fi
  if [[ -n "$output_spec" ]]; then
    status_args+=(--output "$output_spec")
  fi
  "${status_args[@]}" >/dev/null
  manifests+=("$manifest")
  rm -f "$stdout_file" "$stderr_file"
}

BETWARE_FASTA="$ROOT/tools/betaware/example/1qj8.fasta"
BETAWARE_PROFILE="$ROOT/tools/betaware/example/1qj8.prof"
BETAWARE_OUT="$ROOT/runs/betaware_1qj8_smoke.out"
run_method \
  "betaware" \
  "fasta=$BETWARE_FASTA" \
  "output=$BETAWARE_OUT" \
  bash "$ROOT/scripts/run_betaware.sh" "$BETWARE_FASTA" "$BETAWARE_PROFILE" "$BETAWARE_OUT"

JUCH_FASTA="$ROOT/tools/betaware/example/1qj8.fasta"
run_method \
  "pred_tmbb2_hmm" \
  "fasta=$JUCH_FASTA" \
  "output=$ROOT/runs/juchmme_hmm_1qj8_smoke.out" \
  bash "$ROOT/scripts/run_juchmme_pred_tmbb2.sh" hmm

run_method \
  "pred_tmbb2_hnn" \
  "fasta=$JUCH_FASTA" \
  "output=$ROOT/runs/juchmme_hnn_1qj8_smoke.out" \
  bash "$ROOT/scripts/run_juchmme_pred_tmbb2.sh" hnn

PROFTMB_QUERY="$ROOT/tools/proftmb_deb/root/usr/share/doc/proftmb/examples/example.Q"
PROFTMB_OUT="$ROOT/runs/proftmb_smoke/result_proftmb_tabular.txt"
run_method \
  "proftmb" \
  "query=$PROFTMB_QUERY" \
  "tabular=$PROFTMB_OUT" \
  bash "$ROOT/scripts/run_proftmb.sh"

TMBED_FASTA="$ROOT/tools/TMbed/examples/sample.fasta"
TMBED_EMBEDDINGS="$ROOT/tools/TMbed/examples/sample.h5"
TMBED_OUT="$ROOT/runs/tmbed_smoke/sample.pred"
run_method \
  "tmbed" \
  "fasta=$TMBED_FASTA" \
  "predictions=$TMBED_OUT" \
  bash "$ROOT/scripts/run_tmbed.sh" "$TMBED_FASTA" "$TMBED_EMBEDDINGS" "$TMBED_OUT"

POLAR_INPUT="$ROOT/runs/polarbearal_input/1A0S.pdb"
POLAR_STDOUT="${POLARBEARAL_STDOUT:-$ROOT/runs/polarbearal_smoke.stdout}"
run_method \
  "polarbearal3" \
  "pdb=$POLAR_INPUT" \
  "stdout=$POLAR_STDOUT" \
  bash "$ROOT/scripts/run_polarbearal.sh" "$POLAR_INPUT" "$ROOT/runs/polarbearal_smoke"

aggregate_args=(
  "$STATUS_PYTHON" "$STATUS_WRITER" aggregate
  --out "$STATUS_DIR/smoke_status.json"
  --run-name "beta_barrel_staves_external_smoke"
  --status "$([[ "$overall_status" -eq 0 ]] && echo ok || echo error)"
  --command-json "$(argv_json bash "$ROOT/scripts/run_smoke_tests.sh")"
  --metric "method_count=${#manifests[@]}"
)
for manifest in "${manifests[@]}"; do
  aggregate_args+=(--method-manifest "$manifest")
done
"${aggregate_args[@]}"

if [[ "$overall_status" -eq 0 ]]; then
  echo "All implemented external baseline smoke tests passed."
fi
exit "$overall_status"
