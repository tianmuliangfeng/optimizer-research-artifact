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

# Experiment 46 / MDP-05. Sync the script directory and this command file.
# Modes:
#   bash commands/46_mdp05_confirmatory_update_shock/20260804_ex46_mdp05_confirmatory_update_shock.sh check
#   bash commands/46_mdp05_confirmatory_update_shock/20260804_ex46_mdp05_confirmatory_update_shock.sh dry-run
#   bash commands/46_mdp05_confirmatory_update_shock/20260804_ex46_mdp05_confirmatory_update_shock.sh pilot
#   MDP05_PILOT_CERTIFICATE=/absolute/.../pilot_precision_certificate.json bash commands/46_mdp05_confirmatory_update_shock/20260804_ex46_mdp05_confirmatory_update_shock.sh formal
#   MDP05_RUN_DIR=/absolute/result MDP05_PILOT_CERTIFICATE=/absolute/.../pilot_precision_certificate.json bash commands/46_mdp05_confirmatory_update_shock/20260804_ex46_mdp05_confirmatory_update_shock.sh resume
#   MDP05_RUN_DIR=/absolute/result bash commands/46_mdp05_confirmatory_update_shock/20260804_ex46_mdp05_confirmatory_update_shock.sh verify
set -euo pipefail

REPO=${SNM_REPO}
RESULTS=${SNM_RESULTS_ROOT}
CTRL_PY=${SNM_CONTROLLER_PYTHON}
CHILD_PY=${SNM_TRAINING_PYTHON}
SOURCE_RUN=${RESULTS}/37_mech09_downproj_refresh_mediation/20260728T075907+0000
SCRIPT_DIR=${REPO}/scripts/46_mdp05_confirmatory_update_shock
OUTPUT_ROOT=${RESULTS}/_shared/analysis/method_deepening_mdp05_confirmatory_update_shock
PILOT_ROOT=${OUTPUT_ROOT}/_pilots
LOCK_ROOT=${SNM_LOCK_ROOT}

MODE=${1:-}
read -r -a GPUS <<< "${MDP05_GPUS:-0 1}"
MAX_PARALLEL=${MDP05_MAX_PARALLEL:-2}

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1

usage() {
  cat <<'EOF'
Usage:
  bash commands/46_mdp05_confirmatory_update_shock/20260804_ex46_mdp05_confirmatory_update_shock.sh check
  bash commands/46_mdp05_confirmatory_update_shock/20260804_ex46_mdp05_confirmatory_update_shock.sh dry-run
  bash commands/46_mdp05_confirmatory_update_shock/20260804_ex46_mdp05_confirmatory_update_shock.sh pilot
  MDP05_PILOT_CERTIFICATE=/absolute/pilot_precision_certificate.json bash commands/46_mdp05_confirmatory_update_shock/20260804_ex46_mdp05_confirmatory_update_shock.sh formal
  MDP05_RUN_DIR=/absolute/formal_result MDP05_PILOT_CERTIFICATE=/absolute/pilot_precision_certificate.json bash commands/46_mdp05_confirmatory_update_shock/20260804_ex46_mdp05_confirmatory_update_shock.sh resume
  MDP05_RUN_DIR=/absolute/formal_result bash commands/46_mdp05_confirmatory_update_shock/20260804_ex46_mdp05_confirmatory_update_shock.sh verify
EOF
}

require_layout() {
  test -d "${SCRIPT_DIR}" || { echo "Missing ${SCRIPT_DIR}"; exit 2; }
  test -f "${SCRIPT_DIR}/run_mdp05.py" || { echo "Missing MDP-05 controller"; exit 2; }
  test -f "${SCRIPT_DIR}/mdp05_contract.json" || { echo "Missing MDP-05 contract"; exit 2; }
  test -d "${SOURCE_RUN}" || { echo "Missing accepted experiment-37 run: ${SOURCE_RUN}"; exit 2; }
  test -x "${CTRL_PY}" || { echo "Missing controller Python: ${CTRL_PY}"; exit 2; }
  test -x "${CHILD_PY}" || { echo "Missing training Python: ${CHILD_PY}"; exit 2; }
}

require_two_gpus() {
  if [[ ${#GPUS[@]} -ne 2 ]]; then
    echo "MDP-05 formal requires exactly two GPU ids; got: ${GPUS[*]}" >&2
    exit 2
  fi
  if [[ "${GPUS[0]}" == "${GPUS[1]}" ]]; then
    echo "MDP-05 GPU ids must be unique" >&2
    exit 2
  fi
}

assert_gpu_idle() {
  local gpu=$1
  local active
  active=$(nvidia-smi -i "${gpu}" --query-compute-apps=pid,process_name \
    --format=csv,noheader,nounits 2>/dev/null || true)
  if [[ -n "${active}" ]]; then
    echo "GPU ${gpu} has an active compute process:" >&2
    echo "${active}" >&2
    exit 73
  fi
}

controller_command() {
  local run_dir=$1
  shift
  "${CTRL_PY}" "${SCRIPT_DIR}/run_mdp05.py" \
    --run-dir "${run_dir}" \
    --source-run "${SOURCE_RUN}" \
    --child-python "${CHILD_PY}" \
    --gpus "${GPUS[@]}" \
    --max-parallel "${MAX_PARALLEL}" \
    "$@"
}

run_check() {
  require_layout
  (
    cd "${SCRIPT_DIR}"
    sha256sum -c SHA256SUMS
  )
  "${CTRL_PY}" "${SCRIPT_DIR}/test_mdp05.py" -v
  "${CTRL_PY}" "${SCRIPT_DIR}/inspect_mdp05.py" --help >/dev/null
  echo "MDP-05 frozen sources and CPU regression tests passed."
}

run_pilot() {
  require_layout
  mkdir -p "${PILOT_ROOT}" "${LOCK_ROOT}"
  local stamp
  local pilot_dir
  stamp=$(date -u +%Y%m%dT%H%M%S+0000)
  pilot_dir=${MDP05_PILOT_DIR:-${PILOT_ROOT}/${stamp}}
  (
    flock -n 8 || { echo "Physical GPU ${GPUS[0]} lock is busy"; exit 73; }
    assert_gpu_idle "${GPUS[0]}"
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${CHILD_PY}" \
      "${SCRIPT_DIR}/pilot_precision.py" \
      --output-dir "${pilot_dir}" \
      --contract "${SCRIPT_DIR}/mdp05_contract.json"
  ) 8>"${LOCK_ROOT}/gpu${GPUS[0]}.lock"
  echo "Use this for formal:"
  echo "export MDP05_PILOT_CERTIFICATE=${pilot_dir}/pilot_precision_certificate.json"
}

run_formal_locked() {
  local run_dir=$1
  shift
  require_two_gpus
  mkdir -p "${LOCK_ROOT}"
  (
    flock -n 8 || { echo "Physical GPU ${GPUS[0]} lock is busy"; exit 73; }
    flock -n 9 || { echo "Physical GPU ${GPUS[1]} lock is busy"; exit 73; }
    assert_gpu_idle "${GPUS[0]}"
    assert_gpu_idle "${GPUS[1]}"
    controller_command "${run_dir}" \
      --pilot-certificate "${MDP05_PILOT_CERTIFICATE}" "$@"
  ) 8>"${LOCK_ROOT}/gpu${GPUS[0]}.lock" 9>"${LOCK_ROOT}/gpu${GPUS[1]}.lock"
}

case "${MODE}" in
  check)
    run_check
    ;;
  dry-run)
    require_layout
    stamp=$(date -u +%Y%m%dT%H%M%S+0000)
    run_dir=${MDP05_RUN_DIR:-${OUTPUT_ROOT}/${stamp}_dryrun}
    echo "MDP05_DRY_RUN=${run_dir}"
    controller_command "${run_dir}" --dry-run
    ;;
  pilot)
    run_pilot
    ;;
  formal)
    require_layout
    : "${MDP05_PILOT_CERTIFICATE:?Run pilot and set MDP05_PILOT_CERTIFICATE}"
    stamp=$(date -u +%Y%m%dT%H%M%S+0000)
    run_dir=${MDP05_RUN_DIR:-${OUTPUT_ROOT}/${stamp}}
    echo "MDP05_RUN_DIR=${run_dir}"
    run_formal_locked "${run_dir}"
    ;;
  resume)
    require_layout
    : "${MDP05_RUN_DIR:?Set MDP05_RUN_DIR to the exact incomplete formal directory}"
    : "${MDP05_PILOT_CERTIFICATE:?Set the same MDP05_PILOT_CERTIFICATE used by formal}"
    echo "MDP05_RUN_DIR=${MDP05_RUN_DIR}"
    run_formal_locked "${MDP05_RUN_DIR}" --resume
    ;;
  verify)
    require_layout
    : "${MDP05_RUN_DIR:?Set MDP05_RUN_DIR to the completed formal directory}"
    "${CTRL_PY}" "${SCRIPT_DIR}/inspect_mdp05.py" --run-dir "${MDP05_RUN_DIR}"
    ;;
  *)
    usage
    exit 2
    ;;
esac
