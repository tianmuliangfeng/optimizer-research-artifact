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

# Run only when both physical GPUs on the R1 node are otherwise idle.
REPO="${REPO:-${SNM_REPO}}"
INPUT_ROOT="${INPUT_ROOT:-${SNM_RESULTS_ROOT}}"
OUT_ROOT="${OUT_ROOT:-${INPUT_ROOT}/39_submission_efficiency_and_sensitivity/followup/r1_isolated_efficiency}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%S+0000)}"
RUN_ROOT="${RUN_ROOT:-${OUT_ROOT}/${STAMP}}"
CTRL_PY="${CTRL_PY:-${SNM_CONTROLLER_PYTHON}}"
TRAIN_PY="${TRAIN_PY:-${SNM_TRAINING_PYTHON}}"
OFFICIAL_REPO="${OFFICIAL_REPO:-${SNM_OFFICIAL_REPO}}"
GPU="${GPU:-0}"
RUNNER="${REPO}/scripts/18_r1_performance/run_r1_performance.py"

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
  | grep -Eq '[0-9]'; then
  echo "Refusing paper-grade timing: at least one GPU compute process is active." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
mkdir -p "${OUT_ROOT}"
exec 8> "${OUT_ROOT}/.node_controller.lock"
if ! flock -n 8; then
  echo "Another experiment-39 efficiency controller is active under ${OUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}"
exec 9> "${RUN_ROOT}/.controller.lock"
if ! flock -n 9; then
  echo "Another isolated-efficiency controller is active for ${RUN_ROOT}" >&2
  exit 2
fi
"${CTRL_PY}" \
  "${REPO}/scripts/39_submission_efficiency_and_sensitivity/certify_exclusive_node.py" \
  --output "${RUN_ROOT}/exclusive_node_preflight.json" \
  --required-gpus 0 1
"${CTRL_PY}" \
  "${REPO}/scripts/39_submission_efficiency_and_sensitivity/certify_official_repo.py" \
  --official-repo "${OFFICIAL_REPO}" \
  --output "${RUN_ROOT}/official_repo_provenance.json"

"${CTRL_PY}" "${RUNNER}" \
  --official-repo "${OFFICIAL_REPO}" \
  --python-exe "${TRAIN_PY}" \
  --output-root "${RUN_ROOT}" \
  --methods diag block4 none muon \
  --preflight

"${CTRL_PY}" "${RUNNER}" \
  --official-repo "${OFFICIAL_REPO}" \
  --python-exe "${TRAIN_PY}" \
  --output-root "${RUN_ROOT}" \
  --methods diag block4 none muon \
  --numerical-smoke

SMOKE_MANIFEST="$(
  find "${RUN_ROOT}" -type f -name perf_manifest.json -path '*_smoke_seed2026/*' \
    -printf '%T@ %p\n' \
  | sort -nr \
  | head -n 1 \
  | cut -d' ' -f2-
)"
test -n "${SMOKE_MANIFEST}"

"${CTRL_PY}" "${RUNNER}" \
  --official-repo "${OFFICIAL_REPO}" \
  --python-exe "${TRAIN_PY}" \
  --output-root "${RUN_ROOT}" \
  --methods diag block4 none muon \
  --smoke-manifest "${SMOKE_MANIFEST}" \
  --training-benchmark \
  --timed-steps 512 \
  --repeats 4

"${CTRL_PY}" \
  "${REPO}/scripts/39_submission_efficiency_and_sensitivity/certify_exclusive_node.py" \
  --output "${RUN_ROOT}/exclusive_node_postflight.json" \
  --required-gpus 0 1

LATEST="$(
  find "${RUN_ROOT}" -type f -name training_benchmark_summary.csv \
    -printf '%T@ %h\n' \
  | sort -nr \
  | head -n 1 \
  | cut -d' ' -f2-
)"
printf '%s\n' "${RUN_ROOT}" > "${OUT_ROOT}/LATEST_RUN_DIR.txt"
echo "R1 isolated efficiency run root:  ${RUN_ROOT}"
echo "R1 isolated efficiency artifacts: ${LATEST}"
echo "R1 isolated efficiency summary:   ${LATEST}/training_benchmark_summary.csv"
