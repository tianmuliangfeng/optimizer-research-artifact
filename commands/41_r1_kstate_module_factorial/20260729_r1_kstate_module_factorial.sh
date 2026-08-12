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
CTRL_PY="${CTRL_PY:-${SNM_CONTROLLER_PYTHON}}"
TRAIN_PY="${TRAIN_PY:-${PYTHON:-${SNM_TRAINING_PYTHON}}}"
OFFICIAL_REPO="${OFFICIAL_REPO:-${SNM_OFFICIAL_REPO}}"
RESULT_ROOT="${RESULT_ROOT:-${SNM_RESULTS_ROOT}/41_r1_kstate_module_factorial}"
EXISTING_SUMMARY="${EXISTING_SUMMARY:-${REPO}/scripts/41_r1_kstate_module_factorial/existing_cells_reference.csv}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%S+0000)}"
RUN_DIR="${RUN_DIR:-${RESULT_ROOT}/${STAMP}}"

if [[ ! -x "${CTRL_PY}" ]]; then
  echo "Controller Python is not executable: ${CTRL_PY}" >&2
  exit 2
fi
if [[ ! -x "${TRAIN_PY}" ]]; then
  echo "Training Python is not executable: ${TRAIN_PY}" >&2
  exit 2
fi
if [[ ! -f "${EXISTING_SUMMARY}" ]]; then
  echo "Frozen experiment-15 summary is missing: ${EXISTING_SUMMARY}" >&2
  exit 2
fi

echo "R1 module factorial run directory: ${RUN_DIR}"
echo "R1 module factorial GPU policy: physical GPUs 0 and 1"
echo "Controller Python: ${CTRL_PY}"
echo "Training Python:   ${TRAIN_PY}"

"${CTRL_PY}" -c 'import sys, wandb; print(f"Controller runtime verified: {sys.executable}; wandb={wandb.__version__}")'
"${CTRL_PY}" "${REPO}/scripts/41_r1_kstate_module_factorial/test_r1_module_factorial.py"

RESUME_ARGS=()
if [[ -d "${RUN_DIR}" ]] && [[ -n "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  RESUME_ARGS+=(--resume)
fi

"${CTRL_PY}" "${REPO}/scripts/41_r1_kstate_module_factorial/run_r1_module_factorial_suite.py" \
  --run-dir "${RUN_DIR}" \
  --repo "${REPO}" \
  --official-repo "${OFFICIAL_REPO}" \
  --python-exe "${TRAIN_PY}" \
  --existing-summary "${EXISTING_SUMMARY}" \
  --gpus 0 1 \
  "${RESUME_ARGS[@]}"

echo "R1 module factorial artifacts: ${RUN_DIR}"
echo "R1 module factorial manifest:  ${RUN_DIR}/r1_module_factorial_manifest.json"
