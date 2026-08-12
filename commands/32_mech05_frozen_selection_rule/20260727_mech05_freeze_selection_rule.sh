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
PYTHON_BIN="${MECH05_PYTHON:-${SNM_CONTROLLER_PYTHON}}"
STAMP="${MECH05_STAMP:-$(date -u +%Y%m%dT%H%M%S+0000)}"
OUTPUT_DIR="${RESULT_ROOT}/32_mech05_frozen_selection_rule/${STAMP}"
export PYTHONDONTWRITEBYTECODE=1

R1_MECH02="${MECH05_R1_MECH02:-${RESULT_ROOT}/30_mech02_k_geometry/r1_native_endpoint/20260727T051725+0000/formal}"
LLAMA_MECH02="${MECH05_LLAMA_MECH02:-${RESULT_ROOT}/30_mech02_k_geometry/llama_host_endpoint/20260727T051732+0000/llama124_endpoint/formal}"
R1_MECH03="${MECH05_R1_MECH03:-${RESULT_ROOT}/31_mech03_crossfit_shadow/r1_native_endpoint/20260727T062154+0000/formal}"
LLAMA_MECH03="${MECH05_LLAMA_MECH03:-${RESULT_ROOT}/31_mech03_crossfit_shadow/llama_host_endpoint/20260727T062207+0000/llama124_endpoint/formal}"
R1_SUMMARY="${MECH05_R1_SUMMARY:-${RESULT_ROOT}/15_official_newton_muon_r1/analysis/wandb_20260721_multiseed_factorial/r1_multiseed_run_summary.csv}"
LLAMA_SUMMARY="${MECH05_LLAMA_SUMMARY:-${RESULT_ROOT}/17_llama_swiglu_validation/analysis/wandb_20260722_multiseed/llama_multiseed_run_summary.csv}"

"${PYTHON_BIN}" \
  "${CODE_ROOT}/scripts/32_mech05_frozen_selection_rule/test_mech05_contract.py"

"${PYTHON_BIN}" \
  "${CODE_ROOT}/scripts/32_mech05_frozen_selection_rule/analyze_mech05.py" \
  --output-dir "${OUTPUT_DIR}" \
  --contract "${CODE_ROOT}/scripts/32_mech05_frozen_selection_rule/selection_rule_contract.json" \
  --r1-mech02-formal-dir "${R1_MECH02}" \
  --llama124-mech02-formal-dir "${LLAMA_MECH02}" \
  --r1-mech03-formal-dir "${R1_MECH03}" \
  --llama124-mech03-formal-dir "${LLAMA_MECH03}" \
  --r1-run-summary "${R1_SUMMARY}" \
  --llama124-run-summary "${LLAMA_SUMMARY}"
