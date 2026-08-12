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
OFFICIAL_REPO="${SNM_OFFICIAL_REPO}"
CTRL_PY="${SNM_CONTROLLER_PYTHON}"
TRAIN_PY="${SNM_TRAINING_PYTHON}"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/24_r1_dense_full_alpha"
RUNNER="${SCRIPT_DIR}/run_r1_dense_full_alpha.py"
CONFIRM_RUNNER="${SCRIPT_DIR}/run_confirmatory_batch.py"
WANDB_PROJECT="Selective-Newton-Muon-R1-DenseFullAlpha-Confirmatory-20260727"

export CUDA_VISIBLE_DEVICES="${R1_DENSE_ALPHA_GPU:-1}"

cd "${PROJECT_ROOT}"
"${CTRL_PY}" "${SCRIPT_DIR}/test_r1_dense_full_alpha.py"

WANDB_ENTITY_ARGS=()
if [[ -n "${R1_ALPHA_WANDB_ENTITY:-}" ]]; then
  WANDB_ENTITY_ARGS+=(--wandb-entity "${R1_ALPHA_WANDB_ENTITY}")
fi

CONCURRENT_ARGS=()
if [[ "${R1_ALPHA_CONCURRENT_NODE:-0}" == "1" ]]; then
  CONCURRENT_ARGS+=(
    --concurrent-node-training
    --concurrent-workload "${R1_DENSE_ALPHA_CONCURRENT_WORKLOAD:-block_alpha_other_gpu}"
  )
fi

COMMON=(
  --official-repo "${OFFICIAL_REPO}"
  --python-exe "${TRAIN_PY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-mode online
  --wandb-train-log-every 20
  --wandb-init-timeout 120
  --run-prefix mainconf_r1_dense_full_alpha_confirmatory
  "${WANDB_ENTITY_ARGS[@]}"
  "${CONCURRENT_ARGS[@]}"
)

RESUME_2024="${R1_DENSE_ALPHA_RESUME_2024:-}"
RESUME_2025="${R1_DENSE_ALPHA_RESUME_2025:-}"

if [[ -z "${RESUME_2024}" && -z "${RESUME_2025}" ]]; then
  "${CTRL_PY}" "${CONFIRM_RUNNER}" \
    "${COMMON[@]}" \
    --seeds 2024 2025 \
    --continue-on-error
else
  for SEED in 2024 2025; do
    if [[ "${SEED}" == "2024" ]]; then
      BATCH="${RESUME_2024}"
    else
      BATCH="${RESUME_2025}"
    fi
    if [[ -n "${BATCH}" ]]; then
      "${CTRL_PY}" "${RUNNER}" \
        "${COMMON[@]}" \
        --seed "${SEED}" \
        --confirmatory \
        --methods \
          fullalpha0 fullalpha0p25 fullalpha0p50 fullalpha0p75 fullalpha1 \
        --resume-batch "${BATCH}" \
        --continue-on-error
      "${CTRL_PY}" -c \
        'import json,sys; p=json.load(open(sys.argv[1])); assert p.get("status")=="completed_valid" and p.get("wandb_complete") is True, p' \
        "${BATCH}/r1_manifest.json"
    else
      "${CTRL_PY}" "${CONFIRM_RUNNER}" \
        "${COMMON[@]}" \
        --seeds "${SEED}" \
        --continue-on-error
    fi
  done
fi
