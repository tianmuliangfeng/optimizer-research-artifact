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
OUTPUT_ROOT="${OUTPUT_ROOT:-${INPUT_ROOT}/38_unified_mechanism_synthesis}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%S+0000)}"
PYTHON="${PYTHON:-python}"

"${PYTHON}" "${REPO}/scripts/38_unified_mechanism_synthesis/test_unified_mechanism.py"
"${PYTHON}" "${REPO}/scripts/38_unified_mechanism_synthesis/analyze_unified_mechanism.py" \
  --input-root "${INPUT_ROOT}" \
  --registry "${REPO}/scripts/38_unified_mechanism_synthesis/source_registry.json" \
  --output-dir "${OUTPUT_ROOT}/${STAMP}"
"${PYTHON}" "${REPO}/scripts/38_unified_mechanism_synthesis/validate_unified_mechanism.py" \
  --input-root "${INPUT_ROOT}" \
  --run-dir "${OUTPUT_ROOT}/${STAMP}"

echo "Unified mechanism artifacts: ${OUTPUT_ROOT}/${STAMP}"
echo "Unified mechanism manifest:  ${OUTPUT_ROOT}/${STAMP}/unified_mechanism_manifest.json"
