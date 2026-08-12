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

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCHER="${ROOT}/commands/46_mdp05_confirmatory_update_shock/20260804_ex46_mdp05_confirmatory_update_shock.sh"
RESULTS="${SNM_RESULTS_ROOT:-${ROOT}/runs}"
STAMP="$(date -u +%Y%m%dT%H%M%S+0000)"

export SNM_ARTIFACT_ROOT="${SNM_ARTIFACT_ROOT:-${ROOT}}"
export MDP05_RUN_DIR="${MDP05_RUN_DIR:-${RESULTS}/_shared/analysis/method_deepening_mdp05_confirmatory_update_shock/${STAMP}}"
export MDP05_PILOT_DIR="${MDP05_PILOT_DIR:-${RESULTS}/_shared/analysis/method_deepening_mdp05_confirmatory_update_shock/_pilots/${STAMP}}"

bash "${LAUNCHER}" check
bash "${LAUNCHER}" pilot
export MDP05_PILOT_CERTIFICATE="${MDP05_PILOT_CERTIFICATE:-${MDP05_PILOT_DIR}/pilot_precision_certificate.json}"
test -s "${MDP05_PILOT_CERTIFICATE}" || {
  echo "Missing MDP-05 pilot certificate: ${MDP05_PILOT_CERTIFICATE}" >&2
  exit 2
}
bash "${LAUNCHER}" formal
bash "${LAUNCHER}" verify

