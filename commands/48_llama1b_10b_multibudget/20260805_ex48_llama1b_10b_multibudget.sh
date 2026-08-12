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

# Experiment 48: LLaMA-1B, four methods x three seeds, three equal-cooldown budgets.

MODE="${1:-check}"
REPO="${REPO:-${SNM_REPO}}"
CONTROLLER_PYTHON="${EX48_CONTROLLER_PYTHON:-${SNM_CONTROLLER_PYTHON}}"
TRAINING_PYTHON="${EX48_TRAINING_PYTHON:-${SNM_TRAINING_PYTHON}}"
# Code and data roots remain independently overridable, but the certificate-safe
# r0 checkout is the default for both on the accepted remote host.
OFFICIAL_REPO="${EX48_OFFICIAL_REPO:-${SNM_OFFICIAL_REPO}}"
DATA_DIR="${EX48_DATA_DIR:-${SNM_OFFICIAL_REPO}/data/fineweb10B}"
ARTIFACT_ROOT="${EX48_ARTIFACT_ROOT:-${SNM_RESULTS_ROOT}/48_llama1b_10b_multibudget}"
CONTROLLER="$REPO/scripts/48_llama1b_10b_multibudget/run_formal.py"
WANDB_MODE="${EX48_WANDB_MODE:-online}"
WANDB_PROJECT="${EX48_WANDB_PROJECT:-Selective-Newton-Muon-MainConf-LLaMA-1B-10B-Formal-20260805}"
read -r -a GPUS <<< "${EX48_GPUS:-0 1 2 3}"

echo "EX48_CONTROLLER_PYTHON=$CONTROLLER_PYTHON"
echo "EX48_TRAINING_PYTHON=$TRAINING_PYTHON"
echo "EX48_OFFICIAL_REPO=$OFFICIAL_REPO"
echo "EX48_DATA_DIR=$DATA_DIR"
echo "EX48_GPUS=${GPUS[*]}"

common_args=(
  --live-repo "$REPO"
  --official-repo "$OFFICIAL_REPO"
  --data-dir "$DATA_DIR"
  --training-python "$TRAINING_PYTHON"
  --gpus "${GPUS[@]}"
  --wandb-mode "$WANDB_MODE"
  --wandb-project "$WANDB_PROJECT"
)

if [[ -n "${EX48_WANDB_ENTITY:-}" ]]; then
  common_args+=(--wandb-entity "$EX48_WANDB_ENTITY")
fi

case "$MODE" in
  check)
    env PYTHONDONTWRITEBYTECODE=1 "$CONTROLLER_PYTHON" "$CONTROLLER" check \
      --live-repo "$REPO"
    ;;
  preflight)
    EX48_RUN_DIR="${EX48_RUN_DIR:-$ARTIFACT_ROOT/$(date -u +%Y%m%dT%H%M%S+0000)}"
    env PYTHONDONTWRITEBYTECODE=1 "$CONTROLLER_PYTHON" "$CONTROLLER" preflight \
      --run-dir "$EX48_RUN_DIR" "${common_args[@]}"
    echo "EX48_RUN_DIR=$EX48_RUN_DIR"
    ;;
  pilot|formal|resume)
    EX48_RUN_DIR="${EX48_RUN_DIR:?export EX48_RUN_DIR from the passed preflight}"
    env PYTHONDONTWRITEBYTECODE=1 "$CONTROLLER_PYTHON" "$CONTROLLER" "$MODE" \
      --run-dir "$EX48_RUN_DIR" "${common_args[@]}"
    echo "EX48_ARTIFACTS=$EX48_RUN_DIR"
    ;;
  upload)
    EX48_RUN_DIR="${EX48_RUN_DIR:?set EX48_RUN_DIR to the formal run directory}"
    upload_args=(
      --run-dir "$EX48_RUN_DIR"
      --wandb-mode "$WANDB_MODE"
      --wandb-project "$WANDB_PROJECT"
    )
    if [[ -n "${EX48_WANDB_ENTITY:-}" ]]; then
      upload_args+=(--wandb-entity "$EX48_WANDB_ENTITY")
    fi
    env PYTHONDONTWRITEBYTECODE=1 "$CONTROLLER_PYTHON" "$CONTROLLER" upload \
      "${upload_args[@]}"
    ;;
  verify)
    EX48_RUN_DIR="${EX48_RUN_DIR:?set EX48_RUN_DIR to the completed formal run directory}"
    env PYTHONDONTWRITEBYTECODE=1 "$CONTROLLER_PYTHON" "$CONTROLLER" verify \
      --run-dir "$EX48_RUN_DIR"
    ;;
  *)
    echo "usage: $0 {check|preflight|pilot|formal|resume|upload|verify}" >&2
    exit 2
    ;;
esac
