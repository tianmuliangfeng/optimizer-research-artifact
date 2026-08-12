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

set -Eeuo pipefail

# Experiment 43: Modded-NanoGPT ~275M, upstream Newton-Muon-2 near Record #28.
# Default to physical GPU0 so experiment 44 may independently use GPU1.

REPO="${REPO:-${SNM_REPO}}"
INPUT_ROOT="${INPUT_ROOT:-${SNM_RESULTS_ROOT}}"
EXPERIMENT_ROOT="${INPUT_ROOT}/43_newton_muon_record28_275m"
OFFICIAL_REPO="${OFFICIAL_REPO:-${SNM_OFFICIAL_REPO}}"
DATA_REPO_ROOT="${DATA_REPO_ROOT:-${OFFICIAL_REPO}}"
CTRL_PY="${CTRL_PY:-${SNM_CONTROLLER_PYTHON}}"
TRAIN_PY="${TRAIN_PY:-${SNM_TRAINING_PYTHON}}"
EXP43_GPUS="${EXP43_GPUS:-0}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-selective-newton-muon}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_UPLOAD_TIMEOUT_SECONDS="${WANDB_UPLOAD_TIMEOUT_SECONDS:-120}"
export PYTHONDONTWRITEBYTECODE=1

if [[ ! -x "${CTRL_PY}" ]]; then
  echo "Controller Python is not executable: ${CTRL_PY}" >&2
  exit 2
fi
if [[ ! -x "${TRAIN_PY}" ]]; then
  echo "Training Python is not executable: ${TRAIN_PY}" >&2
  exit 2
fi
if [[ ! -d "${OFFICIAL_REPO}" ]]; then
  echo "Pinned official-r0 checkout is missing: ${OFFICIAL_REPO}" >&2
  exit 2
fi

mkdir -p "${EXPERIMENT_ROOT}"
RUN_WAS_PROVIDED=0
if [[ -n "${RUN_DIR:-}" ]]; then
  RUN_WAS_PROVIDED=1
else
  RUN_DIR="${EXPERIMENT_ROOT}/$(date -u +%Y%m%dT%H%M%S+0000)"
fi
case "${RUN_DIR}" in
  "${EXPERIMENT_ROOT}"/*) ;;
  *)
    echo "RUN_DIR must stay under ${EXPERIMENT_ROOT}: ${RUN_DIR}" >&2
    exit 2
    ;;
esac
mkdir -p "${RUN_DIR}"

LIVE_SUITE_SCRIPT="${REPO}/scripts/43_newton_muon_record28_275m/run_record28_suite.py"
SNAPSHOT_SUITE_SCRIPT="${RUN_DIR}/source_snapshot/controller/run_record28_suite.py"
SNAPSHOT_MANIFEST="${RUN_DIR}/source_snapshot/source_snapshot_manifest.json"
SUITE_SCRIPT="${LIVE_SUITE_SCRIPT}"
SNAPSHOT_ARGS=()
if [[ -f "${SNAPSHOT_SUITE_SCRIPT}" && -f "${SNAPSHOT_MANIFEST}" ]]; then
  # Recovery must not depend on a subsequently edited or broken live checkout.
  SUITE_SCRIPT="${SNAPSHOT_SUITE_SCRIPT}"
  SNAPSHOT_ARGS=(--snapshot-active --resume)
elif [[ "${RUN_WAS_PROVIDED}" -eq 1 ]]; then
  SNAPSHOT_ARGS=(--resume)
fi

GPU_SPEC="${EXP43_GPUS//,/ }"
read -r -a GPU_ARGS <<< "${GPU_SPEC}"
if [[ "${#GPU_ARGS[@]}" -lt 1 || "${#GPU_ARGS[@]}" -gt 2 ]]; then
  echo "EXP43_GPUS must specify one or two unique physical GPU indices: ${EXP43_GPUS}" >&2
  exit 2
fi
if [[ "${#GPU_ARGS[@]}" -eq 2 && "${GPU_ARGS[0]}" == "${GPU_ARGS[1]}" ]]; then
  echo "EXP43_GPUS contains a duplicate lane: ${EXP43_GPUS}" >&2
  exit 2
fi
for gpu in "${GPU_ARGS[@]}"; do
  if [[ ! "${gpu}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "EXP43_GPUS must use canonical physical indices, not UUID aliases: ${EXP43_GPUS}" >&2
    exit 2
  fi
done

recovery_command() {
  local recovery
  if [[ -f "${SNAPSHOT_SUITE_SCRIPT}" && -f "${SNAPSHOT_MANIFEST}" ]]; then
    recovery=(
      env
      PYTHONDONTWRITEBYTECODE=1
      "${CTRL_PY}"
      "${SNAPSHOT_SUITE_SCRIPT}"
      --run-dir "${RUN_DIR}"
      --live-repo "${REPO}"
      --official-repo "${OFFICIAL_REPO}"
      --data-repo-root "${DATA_REPO_ROOT}"
      --training-python "${TRAIN_PY}"
      --gpus "${GPU_ARGS[@]}"
      --wandb-mode "${WANDB_MODE}"
      --wandb-project "${WANDB_PROJECT}"
      --wandb-upload-timeout-seconds "${WANDB_UPLOAD_TIMEOUT_SECONDS}"
      --snapshot-active
      --resume
    )
    if [[ -n "${WANDB_ENTITY}" ]]; then
      recovery+=(--wandb-entity "${WANDB_ENTITY}")
    fi
  else
    recovery=(
      env
      "RUN_DIR=${RUN_DIR}"
      "EXP43_GPUS=${EXP43_GPUS}"
      "WANDB_MODE=${WANDB_MODE}"
      "WANDB_PROJECT=${WANDB_PROJECT}"
      "WANDB_ENTITY=${WANDB_ENTITY}"
      "WANDB_UPLOAD_TIMEOUT_SECONDS=${WANDB_UPLOAD_TIMEOUT_SECONDS}"
      "OFFICIAL_REPO=${OFFICIAL_REPO}"
      "DATA_REPO_ROOT=${DATA_REPO_ROOT}"
      "CTRL_PY=${CTRL_PY}"
      "TRAIN_PY=${TRAIN_PY}"
      bash
      "${REPO}/commands/43_newton_muon_record28_275m/20260730_newton_muon_record28_275m.sh"
    )
  fi
  printf '%q ' "${recovery[@]}"
}

on_exit() {
  code=$?
  if [[ "${code}" -ne 0 ]]; then
    echo "Experiment 43 stopped with exit code ${code}." >&2
    echo "Recovery command (completed cells are verified and skipped; only an interrupted cell restarts from step 0):" >&2
    recovery_command >&2
    printf '\n' >&2
  fi
}
trap on_exit EXIT

COMMAND=(
  "${CTRL_PY}"
  "${SUITE_SCRIPT}"
  --run-dir "${RUN_DIR}"
  --live-repo "${REPO}"
  --official-repo "${OFFICIAL_REPO}"
  --data-repo-root "${DATA_REPO_ROOT}"
  --training-python "${TRAIN_PY}"
  --gpus "${GPU_ARGS[@]}"
  --wandb-mode "${WANDB_MODE}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-upload-timeout-seconds "${WANDB_UPLOAD_TIMEOUT_SECONDS}"
)
if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMAND+=(--wandb-entity "${WANDB_ENTITY}")
fi
COMMAND+=("${SNAPSHOT_ARGS[@]}")

echo "Experiment 43 controller started"
echo "RUN_DIR=${RUN_DIR}"
echo "OFFICIAL_REPO=${OFFICIAL_REPO}"
echo "DATA_REPO_ROOT=${DATA_REPO_ROOT}"
echo "physical GPU lanes=${GPU_ARGS[*]}"
echo "controller Python=${CTRL_PY}"
echo "training Python=${TRAIN_PY}"
echo "suite controller=${SUITE_SCRIPT}"
echo "W&B upload timeout=${WANDB_UPLOAD_TIMEOUT_SECONDS}s"
echo "quality timing eligible=false"
printf 'command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

"${COMMAND[@]}"

echo "Experiment 43 completed."
echo "Artifacts: ${RUN_DIR}"
echo "Manifest: ${RUN_DIR}/record28_suite_manifest.json"
printf 'Recovery/re-upload command: '
recovery_command
printf '\n'
