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

STAGE="${1:-}"
case "${STAGE}" in
  preflight|pilot|formal|verify|all) ;;
  *)
    echo "Usage: bash commands/49_r1_malt_strong_baseline/20260807_ex49_r1_malt_strong_baseline.sh {preflight|pilot|formal|verify|all}" >&2
    exit 2
    ;;
esac

REPO="${REPO:-${SNM_REPO}}"
CONTROLLER_PYTHON="${EX49_CONTROLLER_PYTHON:-${CONTROLLER_PYTHON:-${SNM_CONTROLLER_PYTHON}}}"
TRAINING_PYTHON="${EX49_TRAINING_PYTHON:-${TRAINING_PYTHON:-${SNM_TRAINING_PYTHON}}}"
OFFICIAL_REPO="${EX49_OFFICIAL_REPO:-${OFFICIAL_REPO:-${SNM_OFFICIAL_REPO}}}"
RESULT_ROOT="${EX49_RESULT_ROOT:-${RESULT_ROOT:-${SNM_RESULTS_ROOT}/49_r1_malt_strong_baseline}}"
if [[ -z "${EX49_RUN_DIR:-}" ]]; then
  echo "Set and export one persistent EX49_RUN_DIR before every stage." >&2
  echo "Example: export EX49_RUN_DIR=${RESULT_ROOT}/\$(date -u +%Y%m%dT%H%M%S+0000)" >&2
  exit 2
fi
EX49_GPUS="${EX49_GPUS:-0 1}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
EX45_ANALYSIS_ROOT="${EX45_ANALYSIS_ROOT:-${REPO}/scripts/49_r1_malt_strong_baseline/historical_control_snapshot}"
EX45_SUMMARY="${EX45_SUMMARY:-${EX45_ANALYSIS_ROOT}/r1_unified_eight_method_run_summary.csv}"
EX45_ANALYSIS_MANIFEST="${EX45_ANALYSIS_MANIFEST:-${EX45_ANALYSIS_ROOT}/analysis_manifest.json}"

if [[ ! -x "${CONTROLLER_PYTHON}" ]]; then
  echo "Controller Python is not executable: ${CONTROLLER_PYTHON}" >&2
  exit 2
fi
if [[ ! -x "${TRAINING_PYTHON}" ]]; then
  echo "Training Python is not executable: ${TRAINING_PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${REPO}/scripts/49_r1_malt_strong_baseline/malt_contract.json" ]]; then
  echo "Experiment-49 scripts were not synchronized under ${REPO}" >&2
  exit 2
fi
if [[ "${STAGE}" == "verify" || "${STAGE}" == "all" ]]; then
  if [[ ! -f "${EX45_SUMMARY}" || ! -f "${EX45_ANALYSIS_MANIFEST}" ]]; then
    echo "Accepted Experiment-45 analysis inputs are missing; set EX45_SUMMARY and EX45_ANALYSIS_MANIFEST." >&2
    exit 2
  fi
fi

read -r -a GPU_ARGS <<< "${EX49_GPUS}"
WANDB_ENTITY_ARGS=()
if [[ -n "${WANDB_ENTITY}" ]]; then
  WANDB_ENTITY_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi
RESUME_ARGS=()
if [[ -d "${EX49_RUN_DIR}" ]] && [[ -n "$(find "${EX49_RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  RESUME_ARGS+=(--resume)
fi

echo "EX49_RUN_DIR=${EX49_RUN_DIR}"
echo "EX49_STAGE=${STAGE}"
echo "EX49_GPUS=${EX49_GPUS}"
echo "EX49_CONTROLLER_PYTHON=${CONTROLLER_PYTHON}"
echo "EX49_TRAINING_PYTHON=${TRAINING_PYTHON}"
echo "EX49_IMPLEMENTATION=paper-derived independent MALT and MALTER-Eq17 R1 adaptations"
echo "EX49_PILOT_PROTOCOL=malt_r1_focused_grid_pilot_v4"
echo "EX49_V4_LINEAGE=fresh 12-cell rerun; no V1/V2/V3 artifact merge"
echo "EX49_MALT_LR_ORDER=0.0160 0.0125 0.0100 0.0090 0.0080 0.0064"
echo "EX49_MALTER_LR_ORDER=0.007 0.009 0.012 0.015 0.018 0.025"
echo "EX49_FORMAL_METHODS=malt malter_eq17"

if [[ "${STAGE}" == "preflight" || "${STAGE}" == "all" ]]; then
  env PYTHONDONTWRITEBYTECODE=1 "${CONTROLLER_PYTHON}" -m unittest discover \
    -s "${REPO}/scripts/49_r1_malt_strong_baseline" \
    -p "test_analyze_*.py" \
    -v
  env PYTHONDONTWRITEBYTECODE=1 "${CONTROLLER_PYTHON}" -m unittest discover \
    -s "${REPO}/scripts/49_r1_malt_strong_baseline" \
    -p "test_run_*.py" \
    -v
  env PYTHONDONTWRITEBYTECODE=1 "${TRAINING_PYTHON}" -m unittest discover \
    -s "${REPO}/scripts/49_r1_malt_strong_baseline" \
    -p "test_malt_optimizer.py" \
    -v
fi

env PYTHONDONTWRITEBYTECODE=1 "${CONTROLLER_PYTHON}" \
  "${REPO}/scripts/49_r1_malt_strong_baseline/run_r1_malt_suite.py" \
  --stage "${STAGE}" \
  --run-dir "${EX49_RUN_DIR}" \
  --repo "${REPO}" \
  --official-repo "${OFFICIAL_REPO}" \
  --training-python "${TRAINING_PYTHON}" \
  --gpus "${GPU_ARGS[@]}" \
  --wandb-mode "${WANDB_MODE}" \
  "${WANDB_ENTITY_ARGS[@]}" \
  --experiment45-summary "${EX45_SUMMARY}" \
  --experiment45-analysis-manifest "${EX45_ANALYSIS_MANIFEST}" \
  "${RESUME_ARGS[@]}"

echo "EX49_ARTIFACTS=${EX49_RUN_DIR}"
