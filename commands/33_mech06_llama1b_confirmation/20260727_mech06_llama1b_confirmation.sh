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

CODE_ROOT="${SNM_REPO}"
RESULT_ROOT="${SNM_RESULTS_ROOT}"
CTRL_PY="${SNM_CONTROLLER_PYTHON}"
TRAIN_PY="${SNM_TRAINING_PYTHON}"
RUNNER="${CODE_ROOT}/scripts/33_mech06_llama1b_confirmation/run_mech06.py"
TESTS="${CODE_ROOT}/scripts/33_mech06_llama1b_confirmation/test_mech06_contract.py"
CONTRACT="${CODE_ROOT}/scripts/33_mech06_llama1b_confirmation/confirmation_contract.json"
MECH05_CONTRACT="${CODE_ROOT}/scripts/32_mech05_frozen_selection_rule/selection_rule_contract.json"
TRITON="${SNM_OFFICIAL_REPO}/triton_kernels.py"
DATA_PATTERN="${SNM_OFFICIAL_REPO}/data/fineweb10B/fineweb_val_*.bin"
MECH01_REFERENCE="${RESULT_ROOT}/27_mech01_unified_k_diagnostics/llama_host_remaining/20260727T045443+0000/llama1b_seed2026_down_none6200/numerical_smoke"

EARLY_RUN="${RESULT_ROOT}/20_llama_swiglu_1b/medium/20260722T034513+0000_formal_seed2026/01_down_none"
LATE_RUN="${RESULT_ROOT}/20_llama_swiglu_1b/formal/gpu0_none_diag/20260723T025322+0000_formal_seed2026/01_down_none"
STAMP="${MECH06_STAMP:-$(date -u +%Y%m%dT%H%M%S+0000)}"
OUTPUT="${RESULT_ROOT}/33_mech06_llama1b_confirmation/${STAMP}"
GPU="${MECH_GPU:-0}"

required=(
  "$RUNNER"
  "$TESTS"
  "$CONTRACT"
  "$MECH05_CONTRACT"
  "$TRITON"
  "$EARLY_RUN/checkpoint_latest.pt"
  "$EARLY_RUN/train_llama_swiglu_base.py"
  "$EARLY_RUN/train_llama_swiglu.py"
  "$LATE_RUN/checkpoint_latest.pt"
  "$LATE_RUN/train_llama_swiglu_base.py"
  "$LATE_RUN/train_llama_swiglu.py"
  "$MECH01_REFERENCE/mech01_manifest.json"
)
for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "MECH-06 missing required file: $path" >&2
    exit 2
  fi
done
if ! compgen -G "$DATA_PATTERN" >/dev/null; then
  echo "MECH-06 data pattern has no matches: $DATA_PATTERN" >&2
  exit 2
fi
if [[ -e "$OUTPUT" ]]; then
  echo "MECH-06 refuses existing output: $OUTPUT" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
"$TRAIN_PY" "$TESTS"

run_with_heartbeat() {
  "$@" &
  local child_pid=$!
  while kill -0 "$child_pid" 2>/dev/null; do
    sleep 30
    if kill -0 "$child_pid" 2>/dev/null; then
      echo "MECH-06 still running: pid=$child_pid utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader || true
    fi
  done
  wait "$child_pid"
}

run_with_heartbeat env CUDA_VISIBLE_DEVICES="$GPU" \
"$CTRL_PY" "$RUNNER" \
  --python-exe "$TRAIN_PY" \
  --output-root "$OUTPUT" \
  --early-checkpoint "$EARLY_RUN/checkpoint_latest.pt" \
  --early-source "$EARLY_RUN/train_llama_swiglu_base.py" \
  --early-profile "$EARLY_RUN/train_llama_swiglu.py" \
  --late-checkpoint "$LATE_RUN/checkpoint_latest.pt" \
  --late-source "$LATE_RUN/train_llama_swiglu_base.py" \
  --late-profile "$LATE_RUN/train_llama_swiglu.py" \
  --triton-kernels "$TRITON" \
  --mech01-reference-smoke-dir "$MECH01_REFERENCE" \
  --confirmation-contract "$CONTRACT" \
  --mech05-contract "$MECH05_CONTRACT" \
  --data-pattern "$DATA_PATTERN" \
  --host-id "llama-host-h100" \
  --execution-domain "llama-host-llama1b"
