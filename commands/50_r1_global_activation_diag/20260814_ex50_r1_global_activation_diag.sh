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
if [[ ! "${STAGE}" =~ ^(preflight|pilot|formal|verify|all)$ ]]; then
  echo "Usage: bash commands/50_r1_global_activation_diag/20260814_ex50_r1_global_activation_diag.sh {preflight|pilot|formal|verify|all}" >&2
  exit 2
fi

EX50_REPO="${EX50_REPO:-${SNM_REPO}}"
EX50_CONTROLLER_PYTHON="${EX50_CONTROLLER_PYTHON:-${SNM_CONTROLLER_PYTHON}}"
EX50_TRAINING_PYTHON="${EX50_TRAINING_PYTHON:-${SNM_TRAINING_PYTHON}}"
EX50_OFFICIAL_REPO="${EX50_OFFICIAL_REPO:-${SNM_OFFICIAL_REPO}}"
EX50_RESULT_ROOT="${EX50_RESULT_ROOT:-${SNM_RESULTS_ROOT}/50_r1_global_activation_diag}"
EX50_GPUS="${EX50_GPUS:-0 1}"
EX50_WANDB_MODE="${EX50_WANDB_MODE:-disabled}"
EX50_WANDB_PROJECT="${EX50_WANDB_PROJECT:-anonymous-optimizer-artifact-ex50}"
EX50_RUN_DIR="${EX50_RUN_DIR:-${RUN_DIR:-${EX50_RESULT_ROOT}/$(date -u +%Y%m%dT%H%M%S+0000)}}"
export EX50_OFFICIAL_REPO

read -r -a GPU_ARGS <<< "${EX50_GPUS}"
if [[ "${#GPU_ARGS[@]}" -ne 2 ]] || [[ "${GPU_ARGS[0]}" != "0" ]] || [[ "${GPU_ARGS[1]}" != "1" ]]; then
  echo "Experiment 50 is frozen to EX50_GPUS='0 1'; observed '${EX50_GPUS}'" >&2
  exit 2
fi
if [[ ! -x "${EX50_CONTROLLER_PYTHON}" ]]; then
  echo "Controller Python is not executable: ${EX50_CONTROLLER_PYTHON}" >&2
  exit 2
fi
if [[ ! -x "${EX50_TRAINING_PYTHON}" ]]; then
  echo "Training Python is not executable: ${EX50_TRAINING_PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${EX50_OFFICIAL_REPO}/train_gpt_newton_muon_1.py" ]]; then
  echo "Official R0 repository is incomplete: ${EX50_OFFICIAL_REPO}" >&2
  exit 2
fi

SUITE="${EX50_REPO}/scripts/50_r1_global_activation_diag/run_global_diag_suite.py"
SNAPSHOT_SUITE="${EX50_RUN_DIR}/source_snapshot/scripts/50_r1_global_activation_diag/run_global_diag_suite.py"
if [[ -f "${SNAPSHOT_SUITE}" ]]; then
  SUITE="${SNAPSHOT_SUITE}"
fi

echo "EX50_RUN_DIR=${EX50_RUN_DIR}"
echo "EX50_CONTROLLER_PYTHON=${EX50_CONTROLLER_PYTHON}"
echo "EX50_TRAINING_PYTHON=${EX50_TRAINING_PYTHON}"
echo "EX50_OFFICIAL_REPO=${EX50_OFFICIAL_REPO}"
echo "EX50_GPUS=${EX50_GPUS}"
echo "EX50_WANDB_MODE=${EX50_WANDB_MODE}"

if [[ "${STAGE}" == "preflight" ]] || [[ "${STAGE}" == "all" ]]; then
  "${EX50_CONTROLLER_PYTHON}" -B "${EX50_REPO}/scripts/50_r1_global_activation_diag/test_global_diag_source.py"
  "${EX50_CONTROLLER_PYTHON}" -B "${EX50_REPO}/scripts/50_r1_global_activation_diag/test_analyze_global_diag.py"
  "${EX50_CONTROLLER_PYTHON}" -B "${EX50_REPO}/scripts/50_r1_global_activation_diag/test_global_diag_suite.py"
fi

RESUME_ARGS=()
if [[ -d "${EX50_RUN_DIR}" ]] && [[ -n "$(find "${EX50_RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  RESUME_ARGS+=(--resume)
fi

COMMAND=(
  "${EX50_CONTROLLER_PYTHON}" -B "${SUITE}"
  --stage "${STAGE}"
  --run-dir "${EX50_RUN_DIR}"
  --repo "${EX50_REPO}"
  --official-repo "${EX50_OFFICIAL_REPO}"
  --python-exe "${EX50_TRAINING_PYTHON}"
  --gpus "${GPU_ARGS[@]}"
  --wandb-mode "${EX50_WANDB_MODE}"
  --wandb-project "${EX50_WANDB_PROJECT}"
  "${RESUME_ARGS[@]}"
)
if [[ -n "${EX50_WANDB_ENTITY:-}" ]]; then
  COMMAND+=(--wandb-entity "${EX50_WANDB_ENTITY}")
fi
"${COMMAND[@]}"

echo "EX50_ARTIFACTS=${EX50_RUN_DIR}"
