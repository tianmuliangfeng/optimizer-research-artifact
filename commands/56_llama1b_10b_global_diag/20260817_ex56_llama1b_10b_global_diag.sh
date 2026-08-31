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

# Experiment 56: LLaMA-1B global-diagonal route, three seeds, three budgets.
MODE="${1:-check}"
if [[ ! "$MODE" =~ ^(check|preflight|pilot|formal|resume|verify|upload|all)$ ]]; then
  echo "usage: $0 {check|preflight|pilot|formal|resume|verify|upload|all}" >&2
  exit 2
fi

REPO="${EX56_REPO:-${SNM_REPO}}"
WORKSPACE="${EX56_WORKSPACE:-${SNM_ARTIFACT_ROOT}}"
CONTROLLER_PYTHON="${EX56_CONTROLLER_PYTHON:-${SNM_CONTROLLER_PYTHON}}"
TRAINING_PYTHON="${EX56_TRAINING_PYTHON:-${SNM_TRAINING_PYTHON}}"
OFFICIAL_REPO="${EX56_OFFICIAL_REPO:-${SNM_OFFICIAL_REPO}}"
DATA_DIR="${EX56_DATA_DIR:-${OFFICIAL_REPO}/data/fineweb10B}"
RESULT_ROOT="${EX56_RESULT_ROOT:-${SNM_RESULTS_ROOT}/56_llama1b_10b_global_diag}"
RUN_DIR="${EX56_RUN_DIR:-${RUN_DIR:-${RESULT_ROOT}/$(date -u +%Y%m%dT%H%M%S+0000)}}"
WANDB_MODE="${EX56_WANDB_MODE:-disabled}"
WANDB_PROJECT="${EX56_WANDB_PROJECT:-anonymous-optimizer-artifact-ex56}"
read -r -a GPUS <<< "${EX56_GPUS:-3}"
if [[ "${#GPUS[@]}" -ne 1 || "${GPUS[0]}" != "3" ]]; then
  echo "EX56 is frozen to physical GPU 3 on the four-GPU long-budget host" >&2
  exit 2
fi

LIVE_CONTROLLER="$REPO/scripts/56_llama1b_10b_global_diag/run_formal.py"
common_args=(
  --run-dir "$RUN_DIR"
  --live-repo "$REPO"
  --official-repo "$OFFICIAL_REPO"
  --data-dir "$DATA_DIR"
  --training-python "$TRAINING_PYTHON"
  --gpus "${GPUS[@]}"
  --wandb-mode "$WANDB_MODE"
  --wandb-project "$WANDB_PROJECT"
)
if [[ -n "${EX56_WANDB_ENTITY:-}" ]]; then
  common_args+=(--wandb-entity "$EX56_WANDB_ENTITY")
fi

run_controller() {
  local controller="$LIVE_CONTROLLER"
  local snapshot_controller="$RUN_DIR/source_snapshot/scripts/56_llama1b_10b_global_diag/run_formal.py"
  if [[ -f "$snapshot_controller" ]]; then controller="$snapshot_controller"; fi
  env PYTHONDONTWRITEBYTECODE=1 "$CONTROLLER_PYTHON" -B "$controller" "$@"
}

echo "EX56_RUN_DIR=$RUN_DIR"
echo "EX56_OFFICIAL_REPO=$OFFICIAL_REPO"
echo "EX56_DATA_DIR=$DATA_DIR"
echo "EX56_GPUS=${GPUS[*]}"

case "$MODE" in
  check)
    run_controller check --live-repo "$REPO" --gpus "${GPUS[@]}"
    ;;
  preflight|pilot|formal|resume)
    run_controller "$MODE" "${common_args[@]}"
    ;;
  verify)
    run_controller verify --run-dir "$RUN_DIR" --gpus "${GPUS[@]}"
    ;;
  upload)
    upload_args=(upload --run-dir "$RUN_DIR" --gpus "${GPUS[@]}" --wandb-mode "$WANDB_MODE" --wandb-project "$WANDB_PROJECT")
    if [[ -n "${EX56_WANDB_ENTITY:-}" ]]; then upload_args+=(--wandb-entity "$EX56_WANDB_ENTITY"); fi
    run_controller "${upload_args[@]}"
    ;;
  all)
    run_controller check --live-repo "$REPO" --gpus "${GPUS[@]}"
    if [[ ! -f "$RUN_DIR/preflight_manifest.json" ]]; then
      run_controller preflight "${common_args[@]}"
    fi
    run_controller pilot "${common_args[@]}"
    if [[ -f "$RUN_DIR/suite_plan.json" ]]; then
      run_controller resume "${common_args[@]}"
    else
      run_controller formal "${common_args[@]}"
    fi
    run_controller verify --run-dir "$RUN_DIR" --gpus "${GPUS[@]}"
    ;;
esac

echo "EX56_ARTIFACTS=$RUN_DIR"
