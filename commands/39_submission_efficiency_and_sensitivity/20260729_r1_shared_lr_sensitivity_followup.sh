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
INPUT_ROOT="${INPUT_ROOT:-${SNM_RESULTS_ROOT}}"
ROOT="${ROOT:-${INPUT_ROOT}/39_submission_efficiency_and_sensitivity/followup/r1_shared_lr_sensitivity}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%S+0000)}"
LATEST_FILE="${ROOT}/LATEST_RUN_DIR.txt"
RESUME="${RESUME:-1}"
if [[ -z "${RUN_DIR:-}" ]]; then
  if [[ "${RESUME}" == "1" && -s "${LATEST_FILE}" ]]; then
    RUN_DIR="$(cat "${LATEST_FILE}")"
  else
    RUN_DIR="${ROOT}/${STAMP}"
  fi
fi
CTRL_PY="${CTRL_PY:-${SNM_CONTROLLER_PYTHON}}"
TRAIN_PY="${TRAIN_PY:-${SNM_TRAINING_PYTHON}}"
OFFICIAL_REPO="${OFFICIAL_REPO:-${SNM_OFFICIAL_REPO}}"
PROJECT="${PROJECT:-Selective-Newton-Muon-R1-LRSensitivity-20260729}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
RUNNER="${REPO}/scripts/39_submission_efficiency_and_sensitivity/run_r1_lr_sensitivity.py"
ANALYZER="${REPO}/scripts/39_submission_efficiency_and_sensitivity/analyze_r1_lr_sensitivity.py"
VALIDATOR="${REPO}/scripts/39_submission_efficiency_and_sensitivity/validate_r1_lr_sensitivity.py"
CONTRACT="${REPO}/scripts/39_submission_efficiency_and_sensitivity/lr_sensitivity_contract.json"

if [[ "${GPU0}" == "${GPU1}" ]]; then
  echo "GPU0 and GPU1 must be distinct physical devices." >&2
  exit 2
fi
mkdir -p "${RUN_DIR}"
exec 8> "${ROOT}/.node_controller.lock"
if ! flock -n 8; then
  echo "Another experiment-39 LR controller is active under ${ROOT}" >&2
  exit 2
fi
exec 9> "${RUN_DIR}/.controller.lock"
if ! flock -n 9; then
  echo "Another LR-sensitivity controller is active for ${RUN_DIR}" >&2
  exit 2
fi
printf '%s\n' "${RUN_DIR}" > "${LATEST_FILE}"

PID0=""
PID1=""
terminate_children() {
  trap - INT TERM HUP
  for pid in "${PID0}" "${PID1}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${PID0}" "${PID1}"; do
    if [[ -n "${pid}" ]]; then
      wait "${pid}" 2>/dev/null || true
    fi
  done
  exit 130
}
trap terminate_children INT TERM HUP

# The two lanes use one physical GPU each.  Rerunning this wrapper reuses
# RUN_DIR, so valid cells are skipped and incomplete cells resume.
CUDA_VISIBLE_DEVICES="${GPU0}" "${CTRL_PY}" "${RUNNER}" \
  --repo "${REPO}" \
  --official-repo "${OFFICIAL_REPO}" \
  --training-python "${TRAIN_PY}" \
  --run-dir "${RUN_DIR}" \
  --lane gpu0 \
  --methods diag none \
  --multipliers 0.8 1.0 1.2 \
  --budget-steps 3000 \
  --warmdown-steps 871 \
  --seed 2026 \
  --wandb-project "${PROJECT}" \
  >> "${RUN_DIR}/gpu0_controller.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES="${GPU1}" "${CTRL_PY}" "${RUNNER}" \
  --repo "${REPO}" \
  --official-repo "${OFFICIAL_REPO}" \
  --training-python "${TRAIN_PY}" \
  --run-dir "${RUN_DIR}" \
  --lane gpu1 \
  --methods block4 muon \
  --multipliers 0.8 1.0 1.2 \
  --budget-steps 3000 \
  --warmdown-steps 871 \
  --seed 2026 \
  --wandb-project "${PROJECT}" \
  >> "${RUN_DIR}/gpu1_controller.log" 2>&1 &
PID1=$!

FAILED=0
wait "${PID0}" || FAILED=1
wait "${PID1}" || FAILED=1
PID0=""
PID1=""
if [[ "${FAILED}" -ne 0 ]]; then
  echo "At least one LR-sensitivity lane failed." >&2
  echo "GPU0 log: ${RUN_DIR}/gpu0_controller.log" >&2
  echo "GPU1 log: ${RUN_DIR}/gpu1_controller.log" >&2
  exit 2
fi

if [[ -f "${RUN_DIR}/analysis/lr_sensitivity_manifest.json" ]] \
  && "${CTRL_PY}" "${VALIDATOR}" \
    --output-dir "${RUN_DIR}/analysis" \
    --contract "${CONTRACT}"; then
  echo "Reusing completed LR-sensitivity analysis: ${RUN_DIR}/analysis"
else
  if [[ -d "${RUN_DIR}/analysis" ]]; then
    mv "${RUN_DIR}/analysis" \
      "${RUN_DIR}/analysis.incomplete.$(date -u +%Y%m%dT%H%M%S+0000)"
  fi
  "${CTRL_PY}" "${ANALYZER}" \
    --run-dir "${RUN_DIR}" \
    --contract "${CONTRACT}" \
    --output-dir "${RUN_DIR}/analysis"
fi
"${CTRL_PY}" "${VALIDATOR}" \
  --output-dir "${RUN_DIR}/analysis" \
  --contract "${CONTRACT}"

echo "R1 LR-sensitivity artifacts: ${RUN_DIR}"
echo "R1 LR-sensitivity manifest:  ${RUN_DIR}/analysis/lr_sensitivity_manifest.json"
