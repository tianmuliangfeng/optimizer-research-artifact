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
RUNNER="$PROJECT_ROOT/scripts/31_mech03_crossfit_shadow/run_mech03.py"
PREDICTION_CONTRACT="$PROJECT_ROOT/scripts/31_mech03_crossfit_shadow/prediction_contract.json"
PREDICTION_SHA256="9b56e112797103cfc8c98948850a50ba59d672255f305f8dce2c0f5941a25712"

R1_WORKSPACE="${SNM_RESULTS_ROOT}/15_official_newton_muon_r1/results/20260718T081938+0000_formal_seed2026/mainconf_official_r1_none_seed2026_20260718T081938+0000/workspace"
CHECKPOINT="$R1_WORKSPACE/logs/5617369c-43db-455e-8df3-73a6f864b1e9/state_step006200.pt"
CHECKPOINT_SHA256="5377a818bddc8e33716c838d163df9c04900314c1ec531536fa2ffe8a89aebc1"
SOURCE="$R1_WORKSPACE/train_r1_none.py"
TRITON="$R1_WORKSPACE/triton_kernels.py"
DATA_PATTERN="${SNM_OFFICIAL_REPO}/data/fineweb10B/fineweb_val_*.bin"
MECH01_SMOKE="${SNM_RESULTS_ROOT}/27_mech01_unified_k_diagnostics/r1_native_seed2026_none6200/20260727T030657+0000/numerical_smoke"
MECH02_FORMAL="${SNM_RESULTS_ROOT}/30_mech02_k_geometry/r1_native_endpoint/20260727T051725+0000/formal"

STAMP="$(date -u +%Y%m%dT%H%M%S+0000)"
ROOT="${SNM_RESULTS_ROOT}/31_mech03_crossfit_shadow/r1_native_endpoint/$STAMP"
SMOKE_OUT="$ROOT/smoke"
FORMAL_OUT="$ROOT/formal"

run_with_heartbeat() {
  "$@" &
  local child_pid=$!
  while kill -0 "$child_pid" 2>/dev/null; do
    sleep "$HEARTBEAT_SECONDS"
    if kill -0 "$child_pid" 2>/dev/null; then
      echo "MECH-03 heartbeat $(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$child_pid gpu=$MECH_GPU"
      nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null || true
    fi
  done
  wait "$child_pid"
}

assert_passed() {
  "$CTRL_PY" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("passed") is True, d' \
    "$1"
}

export CUDA_VISIBLE_DEVICES="$MECH_GPU"
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"

echo "MECH-03 prediction contract:"
echo "$PREDICTION_SHA256  $PREDICTION_CONTRACT" | sha256sum -c -

run_with_heartbeat \
  "$CTRL_PY" "$RUNNER" \
  --analysis-tier smoke \
  --python-exe "$TRAIN_PY" \
  --family r1 \
  --method none \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --hash-checkpoint \
  --source-script "$SOURCE" \
  --triton-kernels "$TRITON" \
  --mech01-smoke-dir "$MECH01_SMOKE" \
  --mech02-formal-dir "$MECH02_FORMAL" \
  --prediction-contract "$PREDICTION_CONTRACT" \
  --data-pattern "$DATA_PATTERN" \
  --layers 0 11 \
  --repeats 1 \
  --batches-per-split 2 \
  --repeat-offsets 0 4096 8192 12288 \
  --host-id r1-native-h100 \
  --execution-domain r1-native \
  --output-dir "$SMOKE_OUT" \
  --run-prefix r1_endpoint_smoke

assert_passed "$SMOKE_OUT/mech03_manifest.json"

run_with_heartbeat \
  "$CTRL_PY" "$RUNNER" \
  --analysis-tier formal \
  --smoke-manifest "$SMOKE_OUT/mech03_manifest.json" \
  --python-exe "$TRAIN_PY" \
  --family r1 \
  --method none \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --hash-checkpoint \
  --source-script "$SOURCE" \
  --triton-kernels "$TRITON" \
  --mech01-smoke-dir "$MECH01_SMOKE" \
  --mech02-formal-dir "$MECH02_FORMAL" \
  --prediction-contract "$PREDICTION_CONTRACT" \
  --data-pattern "$DATA_PATTERN" \
  --host-id r1-native-h100 \
  --execution-domain r1-native \
  --output-dir "$FORMAL_OUT" \
  --run-prefix r1_endpoint_formal

assert_passed "$FORMAL_OUT/mech03_manifest.json"

PACKAGE_DIR="${SNM_RESULTS_ROOT}/31_mech03_crossfit_shadow/handoff_packages"
PACKAGE="$PACKAGE_DIR/mech03_r1_endpoint_${STAMP}.tgz"
mkdir -p "$PACKAGE_DIR"
tar -C "$(dirname "$ROOT")" -czf "$PACKAGE" "$(basename "$ROOT")"
echo "MECH-03 R1 endpoint PASS: $FORMAL_OUT/mech03_manifest.json"
echo "MECH-03 R1 handoff: $PACKAGE"
sha256sum "$PACKAGE"
ls -lh "$PACKAGE"
