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

REPO="${REPO:-${SNM_REPO}}"
RESULTS="${RESULTS:-${SNM_RESULTS_ROOT}}"
CONTROLLER_PY="${CONTROLLER_PY:-${SNM_CONTROLLER_PYTHON}}"
CHILD_PY="${CHILD_PY:-${SNM_TRAINING_PYTHON}}"
MECH_GPU="${MECH_GPU:-0}"

SOURCE="${SNM_RESULTS_ROOT}/20_llama_swiglu_1b/medium/20260722T034513+0000_formal_seed2026/01_down_none/train_llama_swiglu_base.py"
PROFILE="${SNM_RESULTS_ROOT}/20_llama_swiglu_1b/medium/20260722T034513+0000_formal_seed2026/01_down_none/train_llama_swiglu.py"
TRITON="${SNM_OFFICIAL_REPO}/triton_kernels.py"
DATA="${SNM_OFFICIAL_REPO}/data/fineweb10B/fineweb_val_*.bin"
CONTRACT="${REPO}/scripts/35_mech07_llama1b_family_contrast/family_contrast_contract.json"
RESUME_ARGS=()
if [[ -n "${MECH07_RESUME_STAMP:-}" ]]; then
  RESUME_ARGS+=(--resume-stamp "${MECH07_RESUME_STAMP}")
fi

"${CONTROLLER_PY}" \
  "${REPO}/scripts/35_mech07_llama1b_family_contrast/test_mech07_contract.py"

"${CONTROLLER_PY}" \
  "${REPO}/scripts/35_mech07_llama1b_family_contrast/run_mech07.py" \
  --output-root "${RESULTS}/35_mech07_llama1b_family_contrast" \
  --contract "${CONTRACT}" \
  --source-script "${SOURCE}" \
  --profile-script "${PROFILE}" \
  --triton-kernels "${TRITON}" \
  --data-pattern "${DATA}" \
  --child-python "${CHILD_PY}" \
  --gpu "${MECH_GPU}" \
  --host-id "llama-host-h100" \
  --execution-domain "llama-host-llama1b-family-contrast" \
  "${RESUME_ARGS[@]}"
