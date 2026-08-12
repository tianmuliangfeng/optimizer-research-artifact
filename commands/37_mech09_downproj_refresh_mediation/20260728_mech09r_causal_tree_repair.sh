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

REPO="${SNM_REPO}"
RESULTS="${SNM_RESULTS_ROOT}"
CONTROLLER_PY="${SNM_CONTROLLER_PYTHON}"
CHILD_PY="${SNM_TRAINING_PYTHON}"

SCRIPT_DIR="${REPO}/scripts/37_mech09_downproj_refresh_mediation"
OUTPUT_ROOT="${RESULTS}/37_mech09_downproj_refresh_mediation"
MECH08_RUN="${RESULTS}/36_mech08_short_horizon_rollout/20260727T102506+0000"
SOURCE="${RESULTS}/20_llama_swiglu_1b/medium/20260722T034513+0000_formal_seed2026/01_down_none/train_llama_swiglu_base.py"
PROFILE="${RESULTS}/20_llama_swiglu_1b/medium/20260722T034513+0000_formal_seed2026/01_down_none/train_llama_swiglu.py"
TRITON="${SNM_OFFICIAL_REPO}/triton_kernels.py"
TRAIN_PATTERN="${SNM_OFFICIAL_REPO}/data/fineweb10B/fineweb_train_*.bin"
VAL_PATTERN="${SNM_OFFICIAL_REPO}/data/fineweb10B/fineweb_val_*.bin"

# Defaults to both LLaMA-host H100s.
read -r -a GPUS <<< "${MECH09R_GPUS:-0 1}"
MAX_PARALLEL="${MECH09R_MAX_PARALLEL:-${#GPUS[@]}}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"${CONTROLLER_PY}" "${SCRIPT_DIR}/test_mech09r_contract.py"
"${CONTROLLER_PY}" "${SCRIPT_DIR}/test_mech09r_controller.py"
"${CHILD_PY}" "${SCRIPT_DIR}/test_mech09r_worker.py"
"${CONTROLLER_PY}" "${SCRIPT_DIR}/test_mech09r_analysis.py"

COMMAND=(
  "${CONTROLLER_PY}"
  "${SCRIPT_DIR}/run_mech09r.py"
  --output-root "${OUTPUT_ROOT}"
  --contract "${SCRIPT_DIR}/refresh_mediation_repair_contract.json"
  --mech08-control-reference "${SCRIPT_DIR}/mech08_control_reference.json"
  --mech08-run-dir "${MECH08_RUN}"
  --source-script "${SOURCE}"
  --profile-script "${PROFILE}"
  --triton-kernels "${TRITON}"
  --train-data-pattern "${TRAIN_PATTERN}"
  --val-data-pattern "${VAL_PATTERN}"
  --child-python "${CHILD_PY}"
  --gpus "${GPUS[@]}"
  --max-parallel "${MAX_PARALLEL}"
  --host-id "llama-host-h100"
  --execution-domain "llama-host-llama1b-mech09r-causal-tree"
)

if [[ -n "${MECH09R_RESUME_STAMP:-}" ]]; then
  COMMAND+=(--resume-stamp "${MECH09R_RESUME_STAMP}")
fi

echo "MECH-09R starting on GPUs: ${GPUS[*]}"
"${COMMAND[@]}"
