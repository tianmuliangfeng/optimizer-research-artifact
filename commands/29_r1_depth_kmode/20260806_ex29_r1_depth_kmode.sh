#!/usr/bin/env bash
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

# Experiment 29: official-R1 depth x c_proj K-mode, 12 cells x 3 seeds.

MODE="${1:-check}"
REPO="${REPO:-${SNM_REPO}}"
CONTROLLER_PYTHON="${EX29_CONTROLLER_PYTHON:-${SNM_CONTROLLER_PYTHON}}"
TRAINING_PYTHON="${EX29_TRAINING_PYTHON:-${SNM_TRAINING_PYTHON}}"
OFFICIAL_REPO="${EX29_OFFICIAL_REPO:-${SNM_OFFICIAL_REPO}}"
RESULTS_DIR="${EX29_RESULTS_DIR:-${SNM_RESULTS_ROOT}/29_r1_depth_kmode/results}"
WANDB_MODE="${EX29_WANDB_MODE:-online}"
WANDB_PROJECT="${EX29_WANDB_PROJECT:-Selective-Newton-Muon-MainConf-R1-Depth-KMode-20260725}"
RUNNER="$REPO/scripts/29_r1_depth_kmode/run_r1_depth_kmode.py"
BATCH="$REPO/scripts/29_r1_depth_kmode/run_three_seed_batch.py"
TESTS="$REPO/scripts/29_r1_depth_kmode/test_r1_depth_kmode.py"
read -r -a GPUS <<< "${EX29_GPUS:-0 1}"

echo "EX29_CONTROLLER_PYTHON=$CONTROLLER_PYTHON"
echo "EX29_TRAINING_PYTHON=$TRAINING_PYTHON"
echo "EX29_OFFICIAL_REPO=$OFFICIAL_REPO"
echo "EX29_RESULTS_DIR=$RESULTS_DIR"
echo "EX29_GPUS=${GPUS[*]}"
echo "EX29_WANDB_PROJECT=$WANDB_PROJECT"

common_runner_args=(
  --official-repo "$OFFICIAL_REPO"
  --python-exe "$TRAINING_PYTHON"
  --wandb-mode "$WANDB_MODE"
  --wandb-project "$WANDB_PROJECT"
)

case "$MODE" in
  check)
    env PYTHONDONTWRITEBYTECODE=1 "$CONTROLLER_PYTHON" "$TESTS"
    "$CONTROLLER_PYTHON" -c 'import sys, wandb; print("controller_python =", sys.executable); print("wandb =", wandb.__version__)'
    "$TRAINING_PYTHON" -c 'import sys, torch, triton, numpy; print("training_python =", sys.executable); print("torch =", torch.__version__); print("torch_cuda =", torch.version.cuda); print("triton =", triton.__version__); print("numpy =", numpy.__version__)'
    ;;
  preflight)
    if [[ "${#GPUS[@]}" -ne 2 ]]; then
      echo "EX29 requires exactly two GPU identifiers; observed: ${GPUS[*]}" >&2
      exit 2
    fi
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
      env PYTHONDONTWRITEBYTECODE=1 "$CONTROLLER_PYTHON" "$RUNNER" \
      "${common_runner_args[@]}" \
      --seed 2024 \
      --preflight
    ;;
  formal)
    if [[ "${#GPUS[@]}" -ne 2 ]]; then
      echo "EX29 requires exactly two GPU identifiers; observed: ${GPUS[*]}" >&2
      exit 2
    fi
    env PYTHONDONTWRITEBYTECODE=1 "$CONTROLLER_PYTHON" "$BATCH" \
      --official-repo "$OFFICIAL_REPO" \
      --python-exe "$TRAINING_PYTHON" \
      --seeds 2024 2025 2026 \
      --devices "${GPUS[@]}" \
      --smoke-steps 34 \
      --results-dir "$RESULTS_DIR" \
      --wandb-mode "$WANDB_MODE" \
      --wandb-project "$WANDB_PROJECT"
    ;;
  dry-run)
    env PYTHONDONTWRITEBYTECODE=1 "$CONTROLLER_PYTHON" "$BATCH" \
      --official-repo "$OFFICIAL_REPO" \
      --python-exe "$TRAINING_PYTHON" \
      --seeds 2024 2025 2026 \
      --devices "${GPUS[@]}" \
      --smoke-steps 34 \
      --results-dir "$RESULTS_DIR" \
      --wandb-mode "$WANDB_MODE" \
      --wandb-project "$WANDB_PROJECT" \
      --dry-run
    ;;
  *)
    echo "usage: $0 {check|dry-run|preflight|formal}" >&2
    exit 2
    ;;
esac
