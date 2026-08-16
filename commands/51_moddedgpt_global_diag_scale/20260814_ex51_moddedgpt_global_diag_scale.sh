#!/usr/bin/env bash
_snm_search_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ "${_snm_search_dir}" != "/" && ! -f "${_snm_search_dir}/pyproject.toml" ]]; do
  _snm_search_dir="$(dirname -- "${_snm_search_dir}")"
done
SNM_ARTIFACT_ROOT="${SNM_ARTIFACT_ROOT:-${_snm_search_dir}}"
SNM_REPO="${SNM_REPO:-${SNM_ARTIFACT_ROOT}}"
SNM_RESULTS_ROOT="${SNM_RESULTS_ROOT:-${SNM_ARTIFACT_ROOT}/runs}"
SNM_OFFICIAL_REPO="${SNM_OFFICIAL_REPO:-${SNM_ARTIFACT_ROOT}/third_party/Newton-Muon-official-r0}"
_snm_default_python="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
SNM_CONTROLLER_PYTHON="${SNM_CONTROLLER_PYTHON:-${_snm_default_python:-python3}}"
SNM_TRAINING_PYTHON="${SNM_TRAINING_PYTHON:-${_snm_default_python:-python3}}"
export SNM_ARTIFACT_ROOT SNM_REPO SNM_RESULTS_ROOT SNM_OFFICIAL_REPO
export SNM_CONTROLLER_PYTHON SNM_TRAINING_PYTHON

set -euo pipefail

STAGE="${1:-}"
if [[ ! "${STAGE}" =~ ^(preflight|pilot|formal|upload|verify|all)$ ]]; then
  echo "Usage: bash commands/51_moddedgpt_global_diag_scale/20260814_ex51_moddedgpt_global_diag_scale.sh {preflight|pilot|formal|upload|verify|all}" >&2
  exit 2
fi

EX51_REPO="${EX51_REPO:-${SNM_REPO}}"
EX51_CONTROLLER_PYTHON="${EX51_CONTROLLER_PYTHON:-${SNM_CONTROLLER_PYTHON}}"
EX51_TRAINING_PYTHON="${EX51_TRAINING_PYTHON:-${SNM_TRAINING_PYTHON}}"
EX51_OFFICIAL_REPO="${EX51_OFFICIAL_REPO:-${SNM_OFFICIAL_REPO}}"
# The data view remains inside r0 and exposes exactly the accepted 50+1 files.
# It never points EX51 at a second official-code checkout.
EX51_DATA_REPO_ROOT="${EX51_DATA_REPO_ROOT:-${EX51_OFFICIAL_REPO}/ex51_frozen50_data_repo}"
EX51_RESULT_ROOT="${EX51_RESULT_ROOT:-${SNM_RESULTS_ROOT}/51_moddedgpt_global_diag_scale}"
EX51_GPUS="${EX51_GPUS:-0 1}"
EX51_WANDB_MODE="${EX51_WANDB_MODE:-disabled}"
EX51_WANDB_PROJECT="${EX51_WANDB_PROJECT:-anonymous-optimizer-artifact-ex51}"
EX51_RUN_DIR="${EX51_RUN_DIR:-${RUN_DIR:-${EX51_RESULT_ROOT}/$(date -u +%Y%m%dT%H%M%S+0000)}}"

read -r -a GPU_ARGS <<< "${EX51_GPUS}"
if [[ "${#GPU_ARGS[@]}" -lt 1 ]] || [[ "${#GPU_ARGS[@]}" -gt 2 ]]; then
  echo "Experiment 51 requires one or two GPU ids; observed '${EX51_GPUS}'" >&2
  exit 2
fi
if [[ ! -x "${EX51_CONTROLLER_PYTHON}" ]] || [[ ! -x "${EX51_TRAINING_PYTHON}" ]]; then
  echo "Experiment 51 Python environment is incomplete" >&2
  exit 2
fi
if [[ ! -f "${EX51_OFFICIAL_REPO}/train_gpt_newton_muon_2.py" ]]; then
  echo "Experiment 51 requires the pinned Newton-Muon-official-r0 checkout: ${EX51_OFFICIAL_REPO}" >&2
  exit 2
fi

if [[ "${STAGE}" == "preflight" ]] || [[ "${STAGE}" == "all" ]]; then
  "${EX51_CONTROLLER_PYTHON}" -B \
    "${EX51_REPO}/scripts/51_moddedgpt_global_diag_scale/prepare_frozen51_data_repo.py" \
    --official-repo "${EX51_OFFICIAL_REPO}" \
    --view-root "${EX51_DATA_REPO_ROOT}"
fi

SUITE="${EX51_REPO}/scripts/51_moddedgpt_global_diag_scale/run_global_diag_scale_suite.py"
SNAPSHOT_SUITE="${EX51_RUN_DIR}/source_snapshot/scripts/51_moddedgpt_global_diag_scale/run_global_diag_scale_suite.py"
if [[ -f "${SNAPSHOT_SUITE}" ]]; then SUITE="${SNAPSHOT_SUITE}"; fi

echo "EX51_RUN_DIR=${EX51_RUN_DIR}"
echo "EX51_OFFICIAL_REPO=${EX51_OFFICIAL_REPO}"
echo "EX51_DATA_REPO_ROOT=${EX51_DATA_REPO_ROOT}"
echo "EX51_GPUS=${EX51_GPUS}"

ARGS=("${EX51_CONTROLLER_PYTHON}" -B "${SUITE}" --stage "${STAGE}" --run-dir "${EX51_RUN_DIR}" --repo "${EX51_REPO}" --official-repo "${EX51_OFFICIAL_REPO}" --data-repo-root "${EX51_DATA_REPO_ROOT}" --training-python "${EX51_TRAINING_PYTHON}" --gpus "${GPU_ARGS[@]}" --wandb-mode "${EX51_WANDB_MODE}" --wandb-project "${EX51_WANDB_PROJECT}")
if [[ -d "${EX51_RUN_DIR}" ]]; then ARGS+=(--resume); fi
if [[ -n "${EX51_WANDB_ENTITY:-}" ]]; then ARGS+=(--wandb-entity "${EX51_WANDB_ENTITY}"); fi
"${ARGS[@]}"
echo "EX51_ARTIFACTS=${EX51_RUN_DIR}"
