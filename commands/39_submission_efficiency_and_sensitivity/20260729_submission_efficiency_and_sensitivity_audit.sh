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
OUTPUT_ROOT="${OUTPUT_ROOT:-${INPUT_ROOT}/39_submission_efficiency_and_sensitivity}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%S+0000)}"
PYTHON="${PYTHON:-python}"
PORTABLE_ROOT="${PORTABLE_ROOT:-${REPO}/scripts/39_submission_efficiency_and_sensitivity/source_snapshot}"
REQUIRE_SUBMISSION_READY="${REQUIRE_SUBMISSION_READY:-0}"

"${PYTHON}" "${REPO}/scripts/39_submission_efficiency_and_sensitivity/test_submission_evidence.py"
"${PYTHON}" "${REPO}/scripts/39_submission_efficiency_and_sensitivity/analyze_submission_evidence.py" \
  --input-root "${INPUT_ROOT}" \
  --registry "${REPO}/scripts/39_submission_efficiency_and_sensitivity/evidence_registry.json" \
  --portable-root "${PORTABLE_ROOT}" \
  --preflight-only
"${PYTHON}" "${REPO}/scripts/39_submission_efficiency_and_sensitivity/analyze_submission_evidence.py" \
  --input-root "${INPUT_ROOT}" \
  --registry "${REPO}/scripts/39_submission_efficiency_and_sensitivity/evidence_registry.json" \
  --portable-root "${PORTABLE_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/${STAMP}"
VALIDATOR_ARGS=(--run-dir "${OUTPUT_ROOT}/${STAMP}")
if [[ "${REQUIRE_SUBMISSION_READY}" == "1" ]]; then
  VALIDATOR_ARGS+=(--require-submission-ready)
fi
"${PYTHON}" "${REPO}/scripts/39_submission_efficiency_and_sensitivity/validate_submission_evidence.py" \
  "${VALIDATOR_ARGS[@]}"
printf '%s\n' "${OUTPUT_ROOT}/${STAMP}" > "${OUTPUT_ROOT}/LATEST_AUDIT_DIR.txt"

echo "Submission evidence artifacts: ${OUTPUT_ROOT}/${STAMP}"
echo "Submission evidence manifest:  ${OUTPUT_ROOT}/${STAMP}/submission_evidence_manifest.json"
