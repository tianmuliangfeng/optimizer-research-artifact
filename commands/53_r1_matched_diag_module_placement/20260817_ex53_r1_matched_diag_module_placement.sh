#!/usr/bin/env bash
set -euo pipefail

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

STAGE="${1:-}"
if [[ ! "${STAGE}" =~ ^(preflight|pilot|formal|verify|all|resume)$ ]]; then
  echo "Usage: bash commands/53_r1_matched_diag_module_placement/20260817_ex53_r1_matched_diag_module_placement.sh {preflight|pilot|formal|verify|all|resume}" >&2
  exit 2
fi

EX53_REPO="${EX53_REPO:-${SNM_REPO}}"
EX53_WORKSPACE="${EX53_WORKSPACE:-${SNM_ARTIFACT_ROOT}}"
EX53_CONTROLLER_PYTHON="${EX53_CONTROLLER_PYTHON:-${SNM_CONTROLLER_PYTHON}}"
EX53_TRAINING_PYTHON="${EX53_TRAINING_PYTHON:-${SNM_TRAINING_PYTHON}}"
EX53_OFFICIAL_REPO="${EX53_OFFICIAL_REPO:-${SNM_OFFICIAL_REPO}}"
EX53_RESULT_ROOT="${EX53_RESULT_ROOT:-${SNM_RESULTS_ROOT}/53_r1_matched_diag_module_placement}"
EX53_GPUS="${EX53_GPUS:-0}"
EX53_WANDB_MODE="${EX53_WANDB_MODE:-disabled}"
EX53_WANDB_PROJECT="${EX53_WANDB_PROJECT:-anonymous-optimizer-artifact-ex53}"
export EX53_OFFICIAL_REPO

if [[ "${STAGE}" == "resume" ]]; then
  if [[ -z "${EX53_RUN_DIR:-${RUN_DIR:-}}" ]]; then
    echo "resume requires EX53_RUN_DIR (or RUN_DIR) to name the existing run" >&2
    exit 2
  fi
  EX53_RUN_DIR="${EX53_RUN_DIR:-${RUN_DIR}}"
else
  EX53_RUN_DIR="${EX53_RUN_DIR:-${RUN_DIR:-${EX53_RESULT_ROOT}/$(date -u +%Y%m%dT%H%M%S+0000)}}"
fi

read -r -a GPU_ARGS <<< "${EX53_GPUS}"
if [[ "${#GPU_ARGS[@]}" -ne 1 ]] || [[ "${GPU_ARGS[0]}" != "0" ]]; then
  echo "Experiment 53 is frozen to EX53_GPUS='0'; observed '${EX53_GPUS}'" >&2
  exit 2
fi
if [[ ! -x "${EX53_CONTROLLER_PYTHON}" ]]; then
  echo "Controller Python is not executable: ${EX53_CONTROLLER_PYTHON}" >&2
  exit 2
fi
if [[ ! -x "${EX53_TRAINING_PYTHON}" ]]; then
  echo "Training Python is not executable: ${EX53_TRAINING_PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${EX53_OFFICIAL_REPO}/train_gpt_newton_muon_1.py" ]] || [[ ! -f "${EX53_OFFICIAL_REPO}/triton_kernels.py" ]]; then
  echo "Official R0 repository is incomplete: ${EX53_OFFICIAL_REPO}" >&2
  exit 2
fi

SUITE="${EX53_REPO}/scripts/53_r1_matched_diag_module_placement/run_matched_diag_suite.py"
SNAPSHOT_SUITE="${EX53_RUN_DIR}/source_snapshot/scripts/53_r1_matched_diag_module_placement/run_matched_diag_suite.py"
if [[ -f "${SNAPSHOT_SUITE}" ]]; then
  SUITE="${SNAPSHOT_SUITE}"
fi

echo "EX53_RUN_DIR=${EX53_RUN_DIR}"
echo "EX53_CONTROLLER_PYTHON=${EX53_CONTROLLER_PYTHON}"
echo "EX53_TRAINING_PYTHON=${EX53_TRAINING_PYTHON}"
echo "EX53_OFFICIAL_REPO=${EX53_OFFICIAL_REPO}"
echo "EX53_GPUS=${EX53_GPUS}"
echo "EX53_WANDB_MODE=${EX53_WANDB_MODE}"

if [[ "${STAGE}" == "preflight" ]] || [[ "${STAGE}" == "all" ]]; then
  "${EX53_CONTROLLER_PYTHON}" -B "${EX53_REPO}/scripts/53_r1_matched_diag_module_placement/test_matched_diag_source.py"
  "${EX53_CONTROLLER_PYTHON}" -B "${EX53_REPO}/scripts/53_r1_matched_diag_module_placement/test_analyze_matched_diag.py"
  "${EX53_CONTROLLER_PYTHON}" -B "${EX53_REPO}/scripts/53_r1_matched_diag_module_placement/test_matched_diag_suite.py"
fi

RESUME_ARGS=()
if [[ -d "${EX53_RUN_DIR}" ]] && [[ -n "$(find "${EX53_RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  RESUME_ARGS+=(--resume)
fi

COMMAND=(
  "${EX53_CONTROLLER_PYTHON}" -B "${SUITE}"
  --stage "${STAGE}"
  --run-dir "${EX53_RUN_DIR}"
  --repo "${EX53_REPO}"
  --official-repo "${EX53_OFFICIAL_REPO}"
  --python-exe "${EX53_TRAINING_PYTHON}"
  --gpus "${GPU_ARGS[@]}"
  --wandb-mode "${EX53_WANDB_MODE}"
  --wandb-project "${EX53_WANDB_PROJECT}"
  "${RESUME_ARGS[@]}"
)
if [[ -n "${EX53_WANDB_ENTITY:-}" ]]; then
  COMMAND+=(--wandb-entity "${EX53_WANDB_ENTITY}")
fi
"${COMMAND[@]}"

echo "EX53_ARTIFACTS=${EX53_RUN_DIR}"
