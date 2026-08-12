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

# One-command, recoverable controller for experiment 39.
REPO="${REPO:-${SNM_REPO}}"
INPUT_ROOT="${INPUT_ROOT:-${SNM_RESULTS_ROOT}}"
CTRL_PY="${CTRL_PY:-${SNM_CONTROLLER_PYTHON}}"
TRAIN_PY="${TRAIN_PY:-${SNM_TRAINING_PYTHON}}"
OFFICIAL_REPO="${OFFICIAL_REPO:-${SNM_OFFICIAL_REPO}}"
PORTABLE_ROOT="${PORTABLE_ROOT:-${REPO}/scripts/39_submission_efficiency_and_sensitivity/source_snapshot}"
EXP_ROOT="${INPUT_ROOT}/39_submission_efficiency_and_sensitivity"
MASTER_LOG="${EXP_ROOT}/experiment39_controller.log"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
RESUME="${RESUME:-1}"
PINNED_OFFICIAL_COMMIT="df78af0db523d8bceb25af4919a3e3e7082b80f3"

OFFICIAL_REPO_INPUT="${OFFICIAL_REPO}"
OFFICIAL_REPO="$(readlink -f "${OFFICIAL_REPO_INPUT}" 2>/dev/null || true)"
if [[ -z "${OFFICIAL_REPO}" || ! -d "${OFFICIAL_REPO}" ]]; then
  echo "Official repository is missing: ${OFFICIAL_REPO_INPUT}" >&2
  exit 2
fi
export REPO INPUT_ROOT CTRL_PY TRAIN_PY OFFICIAL_REPO PORTABLE_ROOT

mkdir -p "${EXP_ROOT}"
exec 8> "${EXP_ROOT}/.experiment39_full.lock"
if ! flock -n 8; then
  echo "Another full experiment-39 controller is active." >&2
  exit 2
fi
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "[$(date -u +%FT%TZ)] experiment 39 controller started"
echo "REPO=${REPO}"
echo "INPUT_ROOT=${INPUT_ROOT}"
echo "GPU lanes=${GPU0},${GPU1}"
echo "RESUME=${RESUME}"

REGISTRY="${REPO}/scripts/39_submission_efficiency_and_sensitivity/evidence_registry.json"
ANALYZER="${REPO}/scripts/39_submission_efficiency_and_sensitivity/analyze_submission_evidence.py"
SNAPSHOT_COUNT="$(
  find "${PORTABLE_ROOT}" -maxdepth 1 -type f \
    \( -name '*.json' -o -name '*.csv' \) | wc -l
)"
if [[ "${SNAPSHOT_COUNT}" -ne 14 ]]; then
  echo "Expected 14 portable evidence files, found ${SNAPSHOT_COUNT}: ${PORTABLE_ROOT}" >&2
  exit 2
fi

echo "[$(date -u +%FT%TZ)] stage 1/5: code and source preflight"
OBSERVED_OFFICIAL_COMMIT="$(
  git -c "safe.directory=${OFFICIAL_REPO}" \
    -C "${OFFICIAL_REPO}" rev-parse HEAD
)"
if [[ "${OBSERVED_OFFICIAL_COMMIT}" != "${PINNED_OFFICIAL_COMMIT}" ]]; then
  echo "Official commit mismatch: expected ${PINNED_OFFICIAL_COMMIT}, observed ${OBSERVED_OFFICIAL_COMMIT}" >&2
  exit 2
fi
TRACKED_CHANGES="$(
  git -c "safe.directory=${OFFICIAL_REPO}" \
    -C "${OFFICIAL_REPO}" status --porcelain --untracked-files=no
)"
if [[ -n "${TRACKED_CHANGES}" ]]; then
  echo "Official repository has tracked changes; refusing experiment 39." >&2
  printf '%s\n' "${TRACKED_CHANGES}" >&2
  exit 2
fi
echo "Git preflight passed: ${OBSERVED_OFFICIAL_COMMIT}"
"${CTRL_PY}" "${REPO}/scripts/39_submission_efficiency_and_sensitivity/test_submission_evidence.py"
"${CTRL_PY}" "${REPO}/scripts/39_submission_efficiency_and_sensitivity/test_r1_lr_sensitivity.py"
"${CTRL_PY}" "${ANALYZER}" \
  --input-root "${INPUT_ROOT}" \
  --registry "${REGISTRY}" \
  --portable-root "${PORTABLE_ROOT}" \
  --preflight-only

echo "[$(date -u +%FT%TZ)] stage 2/5: exclusive two-GPU launch preflight"
if [[ "${GPU0}" == "${GPU1}" ]]; then
  echo "GPU0 and GPU1 must be distinct physical devices." >&2
  exit 2
fi
"${CTRL_PY}" \
  "${REPO}/scripts/39_submission_efficiency_and_sensitivity/certify_exclusive_node.py" \
  --output "${EXP_ROOT}/exclusive_node_before_lr_sensitivity.json" \
  --required-gpus "${GPU0}" "${GPU1}"

echo "[$(date -u +%FT%TZ)] stage 3/5: two-GPU LR sensitivity"
GPU0="${GPU0}" GPU1="${GPU1}" RESUME="${RESUME}" \
  bash "${REPO}/commands/39_submission_efficiency_and_sensitivity/20260729_r1_shared_lr_sensitivity_followup.sh"

echo "[$(date -u +%FT%TZ)] stage 4/5: isolated GPU0 efficiency"
GPU="${GPU0}" \
  bash "${REPO}/commands/39_submission_efficiency_and_sensitivity/20260729_r1_isolated_efficiency_followup.sh"

echo "[$(date -u +%FT%TZ)] stage 5/5: final evidence audit"
PYTHON="${CTRL_PY}" REQUIRE_SUBMISSION_READY=1 \
  bash "${REPO}/commands/39_submission_efficiency_and_sensitivity/20260729_submission_efficiency_and_sensitivity_audit.sh"

echo "[$(date -u +%FT%TZ)] experiment 39 completed successfully"
echo "Controller log: ${MASTER_LOG}"
echo "LR run: $(cat "${EXP_ROOT}/followup/r1_shared_lr_sensitivity/LATEST_RUN_DIR.txt")"
echo "Performance run: $(cat "${EXP_ROOT}/followup/r1_isolated_efficiency/LATEST_RUN_DIR.txt")"
echo "Final audit: $(cat "${EXP_ROOT}/LATEST_AUDIT_DIR.txt")"
