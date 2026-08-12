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

PROJECT_ROOT="${SNM_REPO}"
BLOCK_COMMAND="${PROJECT_ROOT}/commands/22_r1_block_alpha/20260727_r1_block_alpha_confirmatory_multiseed.sh"
DENSE_COMMAND="${PROJECT_ROOT}/commands/24_r1_dense_full_alpha/20260727_r1_dense_full_alpha_confirmatory_multiseed.sh"

cd "${PROJECT_ROOT}"

(
  R1_ALPHA_GPU=0 \
  R1_ALPHA_CONCURRENT_NODE=1 \
  R1_ALPHA_CONCURRENT_WORKLOAD="dense_full_alpha_gpu1" \
  bash "${BLOCK_COMMAND}"
) &
BLOCK_PID=$!

(
  R1_DENSE_ALPHA_GPU=1 \
  R1_ALPHA_CONCURRENT_NODE=1 \
  R1_DENSE_ALPHA_CONCURRENT_WORKLOAD="block_alpha_gpu0" \
  bash "${DENSE_COMMAND}"
) &
DENSE_PID=$!

echo "R1 block-alpha controller PID: ${BLOCK_PID} (physical GPU 0)"
echo "R1 dense-full-alpha controller PID: ${DENSE_PID} (physical GPU 1)"

set +e
wait "${BLOCK_PID}"
BLOCK_STATUS=$?
wait "${DENSE_PID}"
DENSE_STATUS=$?
set -e

echo "R1 block-alpha exit status: ${BLOCK_STATUS}"
echo "R1 dense-full-alpha exit status: ${DENSE_STATUS}"
if [[ "${BLOCK_STATUS}" -ne 0 || "${DENSE_STATUS}" -ne 0 ]]; then
  exit 1
fi
