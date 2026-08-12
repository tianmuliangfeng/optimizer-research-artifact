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

# MDP-04: stream matrix metrics on the accepted experiment-37 LLaMA host.
# Upload scripts/mdp_refresh_streaming and this command file, then use one of:
#   bash commands/mdp04_refresh_streaming/20260803_mdp04_refresh_streaming.sh check
#   bash commands/mdp04_refresh_streaming/20260803_mdp04_refresh_streaming.sh dry-run
#   MDP04_ALLOW_ARCHIVAL_RERUN=1 bash commands/mdp04_refresh_streaming/20260803_mdp04_refresh_streaming.sh formal
#   MDP04_ALLOW_ARCHIVAL_RERUN=1 MDP04_RUN_DIR=/absolute/result/path bash commands/mdp04_refresh_streaming/20260803_mdp04_refresh_streaming.sh resume
#   MDP04_RUN_DIR=/absolute/result/path bash commands/mdp04_refresh_streaming/20260803_mdp04_refresh_streaming.sh verify
#   MDP04_RUN_DIR=/absolute/result/path bash commands/mdp04_refresh_streaming/20260803_mdp04_refresh_streaming.sh archive-verify
set -euo pipefail

REPO=${SNM_REPO}
RESULTS=${SNM_RESULTS_ROOT}
CTRL_PY=${SNM_CONTROLLER_PYTHON}
CHILD_PY=${SNM_TRAINING_PYTHON}
SOURCE_RUN=${RESULTS}/37_mech09_downproj_refresh_mediation/20260728T075907+0000
SCRIPT_DIR=${REPO}/scripts/mdp_refresh_streaming
OUTPUT_ROOT=${RESULTS}/_shared/analysis/method_deepening_mdp04_refresh_replay
LOCK_ROOT=${SNM_LOCK_ROOT}

MODE=${1:-}
read -r -a GPUS <<< "${MDP04_GPUS:-0 1}"
MAX_PARALLEL=${MDP04_MAX_PARALLEL:-2}

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1

usage() {
  cat <<'EOF'
Usage:
  bash commands/mdp04_refresh_streaming/20260803_mdp04_refresh_streaming.sh check
  bash commands/mdp04_refresh_streaming/20260803_mdp04_refresh_streaming.sh dry-run
  bash commands/mdp04_refresh_streaming/20260803_mdp04_refresh_streaming.sh formal
  MDP04_RUN_DIR=/absolute/result/path bash commands/mdp04_refresh_streaming/20260803_mdp04_refresh_streaming.sh resume
  MDP04_RUN_DIR=/absolute/result/path bash commands/mdp04_refresh_streaming/20260803_mdp04_refresh_streaming.sh verify
  MDP04_RUN_DIR=/absolute/result/path bash commands/mdp04_refresh_streaming/20260803_mdp04_refresh_streaming.sh archive-verify

MDP-04 is an archived numeric-gate-failed contract.  A historical formal or
resume execution additionally requires MDP04_ALLOW_ARCHIVAL_RERUN=1 and cannot
upgrade the accepted evidence.  Use archive-verify to reproduce the recorded
negative adjudication without a traceback.
EOF
}

require_code_layout() {
  test -d "${SCRIPT_DIR}" || { echo "Missing ${SCRIPT_DIR}"; exit 2; }
  test -f "${SCRIPT_DIR}/run_stream_replay.py" || { echo "Missing controller"; exit 2; }
  test -f "${SCRIPT_DIR}/inspect_stream_replay.py" || { echo "Missing archive inspector"; exit 2; }
  test -f "${SCRIPT_DIR}/refresh_stream_contract.json" || { echo "Missing contract"; exit 2; }
  test -x "${CTRL_PY}" || { echo "Missing controller Python: ${CTRL_PY}"; exit 2; }
}

require_training_layout() {
  require_code_layout
  test -d "${SOURCE_RUN}" || { echo "Missing accepted experiment-37 run: ${SOURCE_RUN}"; exit 2; }
  test -x "${CHILD_PY}" || { echo "Missing training Python: ${CHILD_PY}"; exit 2; }
}

run_check() {
  require_training_layout
  (
    cd "${SCRIPT_DIR}"
    sha256sum -c SHA256SUMS
  )
  "${CHILD_PY}" "${SCRIPT_DIR}/test_streaming.py" -v
  "${CTRL_PY}" "${SCRIPT_DIR}/test_archive_inspector.py" -v
  "${CTRL_PY}" "${SCRIPT_DIR}/inspect_stream_replay.py" --help >/dev/null
  echo "MDP-04 frozen files, framework tests, and archive inspector passed."
}

require_archival_rerun_acknowledgement() {
  if [[ "${MDP04_ALLOW_ARCHIVAL_RERUN:-0}" != "1" ]]; then
    cat >&2 <<'EOF'
MDP-04 is closed with formal adjudication numeric_gate_failed.
Running formal/resume again cannot upgrade that evidence and is blocked by
default.  For an explicitly authorized historical reproducibility rerun, set:
  MDP04_ALLOW_ARCHIVAL_RERUN=1
For ordinary reproduction of the accepted result, use archive-verify instead.
EOF
    exit 64
  fi
}

controller_command() {
  local run_dir=$1
  shift
  "${CTRL_PY}" "${SCRIPT_DIR}/run_stream_replay.py" \
    --run-dir "${run_dir}" \
    --source-run "${SOURCE_RUN}" \
    --child-python "${CHILD_PY}" \
    --gpus "${GPUS[@]}" \
    --max-parallel "${MAX_PARALLEL}" \
    "$@"
}

assert_gpu_lanes_idle() {
  local gpu
  local active
  for gpu in "${GPUS[@]}"; do
    active=$(nvidia-smi -i "${gpu}" --query-compute-apps=pid,process_name \
      --format=csv,noheader,nounits 2>/dev/null || true)
    if [[ -n "${active}" ]]; then
      echo "GPU ${gpu} has an active compute process:"
      echo "${active}"
      exit 73
    fi
  done
}

run_with_gpu_locks() {
  local run_dir=$1
  shift
  mkdir -p "${LOCK_ROOT}"
  (
    flock -n 8 || { echo "Physical GPU ${GPUS[0]} lock is busy"; exit 73; }
    flock -n 9 || { echo "Physical GPU ${GPUS[1]} lock is busy"; exit 73; }
    assert_gpu_lanes_idle
    controller_command "${run_dir}" "$@"
  ) 8>"${LOCK_ROOT}/gpu${GPUS[0]}.lock" 9>"${LOCK_ROOT}/gpu${GPUS[1]}.lock"
}

case "${MODE}" in
  check)
    run_check
    ;;
  dry-run)
    require_training_layout
    stamp=$(date -u +%Y%m%dT%H%M%S+0000)
    run_dir=${MDP04_RUN_DIR:-${OUTPUT_ROOT}/${stamp}_dryrun}
    echo "MDP04_DRY_RUN=${run_dir}"
    controller_command "${run_dir}" --dry-run
    ;;
  formal)
    require_training_layout
    require_archival_rerun_acknowledgement
    stamp=$(date -u +%Y%m%dT%H%M%S+0000)
    run_dir=${MDP04_RUN_DIR:-${OUTPUT_ROOT}/${stamp}}
    echo "MDP04_RUN_DIR=${run_dir}"
    run_with_gpu_locks "${run_dir}"
    ;;
  resume)
    require_training_layout
    require_archival_rerun_acknowledgement
    : "${MDP04_RUN_DIR:?Set MDP04_RUN_DIR to the exact formal result directory}"
    echo "MDP04_RUN_DIR=${MDP04_RUN_DIR}"
    run_with_gpu_locks "${MDP04_RUN_DIR}" --resume
    ;;
  verify)
    require_code_layout
    : "${MDP04_RUN_DIR:?Set MDP04_RUN_DIR to the exact formal result directory}"
    "${CTRL_PY}" "${SCRIPT_DIR}/inspect_stream_replay.py" \
      --run-dir "${MDP04_RUN_DIR}" --expected passed
    echo "MDP-04 verification passed. Result directory: ${MDP04_RUN_DIR}"
    ;;
  archive-verify)
    require_code_layout
    : "${MDP04_RUN_DIR:?Set MDP04_RUN_DIR to the exact archived result directory}"
    "${CTRL_PY}" "${SCRIPT_DIR}/inspect_stream_replay.py" \
      --run-dir "${MDP04_RUN_DIR}" --expected numeric_gate_failed
    echo "MDP-04 archived adjudication reproduced cleanly: numeric_gate_failed."
    echo "Result directory: ${MDP04_RUN_DIR}"
    ;;
  *)
    usage
    exit 2
    ;;
esac
