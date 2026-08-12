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
SCRIPT_DIR="${PROJECT_ROOT}/scripts/22_r1_block_alpha"
RUNNER="${SCRIPT_DIR}/run_r1_block_alpha.py"
CONFIRM_RUNNER="${SCRIPT_DIR}/run_confirmatory_batch.py"
WANDB_PROJECT="Selective-Newton-Muon-R1-BlockAlpha-Confirmatory-20260724"

# R1 native experiments historically used physical GPU 1. Override explicitly
# if the idle H100 has a different id:
#   R1_ALPHA_GPU=0 bash commands/22_r1_block_alpha/20260727_r1_block_alpha_confirmatory_multiseed.sh
export CUDA_VISIBLE_DEVICES="${R1_ALPHA_GPU:-1}"

cd "${PROJECT_ROOT}"
"${CTRL_PY}" "${SCRIPT_DIR}/test_r1_block_alpha.py"

WANDB_ENTITY_ARGS=()
if [[ -n "${R1_ALPHA_WANDB_ENTITY:-}" ]]; then
  WANDB_ENTITY_ARGS+=(--wandb-entity "${R1_ALPHA_WANDB_ENTITY}")
fi

CONCURRENT_ARGS=()
if [[ "${R1_ALPHA_CONCURRENT_NODE:-0}" == "1" ]]; then
  CONCURRENT_ARGS+=(
    --concurrent-node-training
    --concurrent-workload "${R1_ALPHA_CONCURRENT_WORKLOAD:-dense_full_alpha_other_gpu}"
  )
fi

COMMON=(
  --official-repo "${OFFICIAL_REPO}"
  --python-exe "${TRAIN_PY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-mode online
  --wandb-train-log-every 20
  --wandb-init-timeout 120
  --run-prefix mainconf_r1_block_alpha_confirmatory
  "${WANDB_ENTITY_ARGS[@]}"
  "${CONCURRENT_ARGS[@]}"
)

RESUME_2024="${R1_ALPHA_RESUME_2024:-}"
RESUME_2025="${R1_ALPHA_RESUME_2025:-}"

if [[ -z "${RESUME_2024}" && -z "${RESUME_2025}" ]]; then
  # Fresh unattended launch: per seed, exact-shape four-cell smoke is local
  # only; the four 6200-step formal cells upload after local validation.
  "${CTRL_PY}" "${CONFIRM_RUNNER}" \
    "${COMMON[@]}" \
    --seeds 2024 2025 \
    --continue-on-error
else
  # Recovery mode. Each value must be the exact *_formal_seedYYYY directory,
  # not the outer confirmatory_controller directory. Completed local cells are
  # revalidated and skipped; missing W&B uploads are retried without training.
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
        --methods alpha0 alpha0p25 alpha0p50 alpha0p75 \
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
