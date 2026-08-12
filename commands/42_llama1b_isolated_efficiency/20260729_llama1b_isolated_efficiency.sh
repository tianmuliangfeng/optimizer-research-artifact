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

set -Eeuo pipefail

# Experiment 42: paper-grade, single-host isolated LLaMA-1B efficiency audit.
# GPU0 is the only timing device; GPU1 must remain idle for the entire run.
REPO="${REPO:-${SNM_REPO}}"
INPUT_ROOT="${INPUT_ROOT:-${SNM_RESULTS_ROOT}}"
CTRL_PY="${CTRL_PY:-${SNM_CONTROLLER_PYTHON}}"
TRAIN_PY="${TRAIN_PY:-${SNM_TRAINING_PYTHON}}"
OFFICIAL_REPO="${OFFICIAL_REPO:-${SNM_OFFICIAL_REPO}}"
DATA_DIR="${DATA_DIR:-${OFFICIAL_REPO}/data/fineweb10B}"
SCRIPT_DIR="${REPO}/scripts/42_llama1b_isolated_efficiency"
RESULT_ROOT="${INPUT_ROOT}/42_llama1b_isolated_efficiency"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%S+0000)}"

if [[ -z "${RUN_DIR:-}" ]]; then
  RUN_DIR="${RESULT_ROOT}/${STAMP}"
  RESUME_ARGS=()
else
  RESULT_ROOT_REAL="$(readlink -f "${RESULT_ROOT}")"
  RUN_DIR_REAL="$(readlink -f "${RUN_DIR}" 2>/dev/null || true)"
  if [[ -z "${RUN_DIR_REAL}" || "${RUN_DIR_REAL}" != "${RESULT_ROOT_REAL}/"* ]]; then
    echo "RUN_DIR must be an existing child of ${RESULT_ROOT}" >&2
    exit 2
  fi
  RUN_DIR="${RUN_DIR_REAL}"
  RESUME_ARGS=(--resume)
fi

mkdir -p "${RESULT_ROOT}" "${RUN_DIR}"
print_recovery() {
  local exit_code=$?
  if [[ "${exit_code}" -ne 0 ]]; then
    echo "MECH-42 stopped with exit code ${exit_code}." >&2
    echo "Recovery command:" >&2
    echo "RUN_DIR=\"${RUN_DIR}\" bash \"${REPO}/commands/42_llama1b_isolated_efficiency/20260729_llama1b_isolated_efficiency.sh\"" >&2
  fi
}
trap print_recovery EXIT
exec 8> "${RESULT_ROOT}/.experiment42_node.lock"
if ! flock -n 8; then
  echo "Another experiment-42 controller is active on this LLaMA host." >&2
  exit 2
fi

echo "MECH-42 run directory: ${RUN_DIR}"
echo "MECH-42 policy: GPU0 timed; GPU1 continuously idle; W&B disabled"
echo "MECH-42 stage 1/3: CPU contract tests"
PYTHONPYCACHEPREFIX="${RESULT_ROOT}/.preflight_pycache" \
  "${CTRL_PY}" -m compileall -q "${SCRIPT_DIR}"
PYTHONPYCACHEPREFIX="${RESULT_ROOT}/.preflight_pycache" \
  "${CTRL_PY}" "${SCRIPT_DIR}/test_llama1b_efficiency.py"
PYTHONPYCACHEPREFIX="${RESULT_ROOT}/.preflight_pycache" \
  "${CTRL_PY}" "${SCRIPT_DIR}/test_analyze_llama1b_efficiency.py"

echo "MECH-42 stage 2/3: smoke plus 16 isolated timed cells"
"${CTRL_PY}" "${SCRIPT_DIR}/run_llama1b_efficiency.py" \
  --run-dir "${RUN_DIR}" \
  --official-repo "${OFFICIAL_REPO}" \
  --python-exe "${TRAIN_PY}" \
  --data-dir "${DATA_DIR}" \
  --contract "${SCRIPT_DIR}/efficiency_contract.json" \
  --physical-gpu 0 \
  --required-gpus 0 1 \
  --host-id llama-host-h100 \
  --execution-domain llama-host-llama1b-isolated-efficiency \
  "${RESUME_ARGS[@]}"

echo "MECH-42 stage 3/3: independent aggregate analysis"
"${CTRL_PY}" "${RUN_DIR}/source_snapshot/analyze_llama1b_efficiency.py" \
  --run-dir "${RUN_DIR}" \
  --contract "${RUN_DIR}/efficiency_contract.json" \
  --output-dir "${RUN_DIR}/analysis"

printf '%s\n' "${RUN_DIR}" > "${RESULT_ROOT}/LATEST_RUN_DIR.txt"
echo "MECH-42 artifacts: ${RUN_DIR}"
echo "MECH-42 manifest:  ${RUN_DIR}/analysis/llama1b_efficiency_analysis_manifest.json"
