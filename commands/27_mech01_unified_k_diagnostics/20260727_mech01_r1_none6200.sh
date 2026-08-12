#!/usr/bin/env bash
# Public artifact path profile. Every value may be overridden explicitly.
_snm_search_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ "${_snm_search_dir}" != "/" && ! -f "${_snm_search_dir}/pyproject.toml" ]]; do
  _snm_search_dir="$(dirname -- "${_snm_search_dir}")"
done
SNM_ARTIFACT_ROOT="${SNM_ARTIFACT_ROOT:-${_snm_search_dir}}"
SNM_REPO="${SNM_REPO:-${SNM_ARTIFACT_ROOT}}"
SNM_WORKSPACE_ROOT="${SNM_WORKSPACE_ROOT:-${SNM_ARTIFACT_ROOT}}"
SNM_RESULTS_ROOT="${SNM_RESULTS_ROOT:-${SNM_ARTIFACT_ROOT}/runs}"
SNM_OFFICIAL_REPO="${SNM_OFFICIAL_REPO:-${SNM_ARTIFACT_ROOT}/third_party/Newton-Muon-official-r0}"
_snm_default_python="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
SNM_CONTROLLER_PYTHON="${SNM_CONTROLLER_PYTHON:-${_snm_default_python:-python3}}"
SNM_TRAINING_PYTHON="${SNM_TRAINING_PYTHON:-${_snm_default_python:-python3}}"
SNM_LOCK_ROOT="${SNM_LOCK_ROOT:-${SNM_ARTIFACT_ROOT}/runs/.gpu_locks}"
export SNM_ARTIFACT_ROOT SNM_REPO SNM_WORKSPACE_ROOT SNM_RESULTS_ROOT
export SNM_OFFICIAL_REPO SNM_CONTROLLER_PYTHON SNM_TRAINING_PYTHON SNM_LOCK_ROOT

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SNM_REPO}}"
CTRL_PY="${CTRL_PY:-${SNM_CONTROLLER_PYTHON}}"
TRAIN_PY="${TRAIN_PY:-${SNM_TRAINING_PYTHON}}"
MECH_GPU="${MECH_GPU:-0}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-60}"

RUNNER="$PROJECT_ROOT/scripts/27_mech01_unified_k_diagnostics/run_mech01.py"
R1_WORKSPACE="${SNM_RESULTS_ROOT}/15_official_newton_muon_r1/results/20260718T081938+0000_formal_seed2026/mainconf_official_r1_none_seed2026_20260718T081938+0000/workspace"
R1_CHECKPOINT="$R1_WORKSPACE/logs/5617369c-43db-455e-8df3-73a6f864b1e9/state_step006200.pt"
R1_SOURCE="$R1_WORKSPACE/train_r1_none.py"
R1_TRITON="$R1_WORKSPACE/triton_kernels.py"
R1_DATA_PATTERN="${SNM_OFFICIAL_REPO}/data/fineweb10B/fineweb_val_*.bin"

EXPECTED_CHECKPOINT_BYTES=1312617750
EXPECTED_SOURCE_SHA256="d4368715bb64d6ad89509876c4d7f28773fa4dd567accca09405ff34e74220f5"

STAMP="$(date -u +%Y%m%dT%H%M%S+0000)"
RESULTS_ROOT="${SNM_RESULTS_ROOT}/27_mech01_unified_k_diagnostics/r1_native_seed2026_none6200/$STAMP"
PREFLIGHT_OUT="$RESULTS_ROOT/preflight"
SMOKE_OUT="$RESULTS_ROOT/numerical_smoke"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "MECH-01 missing required file: $1" >&2
    exit 2
  fi
}

run_with_heartbeat() {
  "$@" &
  local child_pid=$!
  while kill -0 "$child_pid" 2>/dev/null; do
    sleep "$HEARTBEAT_SECONDS"
    if kill -0 "$child_pid" 2>/dev/null; then
      echo "MECH-01 heartbeat $(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$child_pid gpu=$MECH_GPU"
      nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null || true
    fi
  done
  wait "$child_pid"
}

require_file "$CTRL_PY"
require_file "$TRAIN_PY"
require_file "$RUNNER"
require_file "$R1_CHECKPOINT"
require_file "$R1_SOURCE"
require_file "$R1_TRITON"

if ! compgen -G "$R1_DATA_PATTERN" >/dev/null; then
  echo "MECH-01 validation shards not found: $R1_DATA_PATTERN" >&2
  exit 2
fi

observed_bytes="$(stat -c '%s' "$R1_CHECKPOINT")"
if [[ "$observed_bytes" != "$EXPECTED_CHECKPOINT_BYTES" ]]; then
  echo "MECH-01 checkpoint size mismatch: observed=$observed_bytes expected=$EXPECTED_CHECKPOINT_BYTES" >&2
  exit 2
fi

observed_source_sha256="$(sha256sum "$R1_SOURCE" | awk '{print $1}')"
if [[ "$observed_source_sha256" != "$EXPECTED_SOURCE_SHA256" ]]; then
  echo "MECH-01 source SHA-256 mismatch: observed=$observed_source_sha256 expected=$EXPECTED_SOURCE_SHA256" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$MECH_GPU"
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"

echo "MECH-01 R1 preflight output: $PREFLIGHT_OUT"
run_with_heartbeat \
  "$CTRL_PY" "$RUNNER" \
  --preflight \
  --python-exe "$TRAIN_PY" \
  --family r1 \
  --checkpoint "$R1_CHECKPOINT" \
  --source-script "$R1_SOURCE" \
  --triton-kernels "$R1_TRITON" \
  --method none \
  --host-id r1-native-h100 \
  --execution-domain r1-native \
  --hash-checkpoint \
  --output-dir "$PREFLIGHT_OUT" \
  --run-prefix r1_none_seed2026_step6200

"$CTRL_PY" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("passed") is True, d' \
  "$PREFLIGHT_OUT/mech01_manifest.json"

CHECKPOINT_SHA256="$(
  "$CTRL_PY" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert d["checkpoint_hash_checked"] and d["checkpoint_hash_pass"]; print(d["checkpoint_sha256_observed"])' \
    "$PREFLIGHT_OUT/checkpoint_schema.json"
)"
echo "MECH-01 checkpoint SHA-256: $CHECKPOINT_SHA256"

echo "MECH-01 R1 numerical smoke output: $SMOKE_OUT"
run_with_heartbeat \
  "$CTRL_PY" "$RUNNER" \
  --numerical-smoke \
  --python-exe "$TRAIN_PY" \
  --family r1 \
  --checkpoint "$R1_CHECKPOINT" \
  --source-script "$R1_SOURCE" \
  --triton-kernels "$R1_TRITON" \
  --method none \
  --data-pattern "$R1_DATA_PATTERN" \
  --layers 0 6 11 \
  --device-batch-size 1 \
  --sequence-length 128 \
  --probe-offsets 0 4096 8192 12288 \
  --max-activation-rows 2048 \
  --ridge-mult 0.2 \
  --ridge-eps 1e-8 \
  --momentum 0.95 \
  --ns-steps 5 \
  --candidates none diag block4 dense_full \
  --spectrum-dtype float64 \
  --export-bundle-layer 6 \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --hash-checkpoint \
  --host-id r1-native-h100 \
  --execution-domain r1-native \
  --output-dir "$SMOKE_OUT" \
  --run-prefix r1_none_seed2026_step6200

"$CTRL_PY" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("passed") is True, d' \
  "$SMOKE_OUT/mech01_manifest.json"

echo "MECH-01 R1 preflight manifest: $PREFLIGHT_OUT/mech01_manifest.json"
echo "MECH-01 R1 smoke manifest:     $SMOKE_OUT/mech01_manifest.json"
echo "MECH-01 fixed tensor bundle:   $SMOKE_OUT/tensor_bundle.pt"
