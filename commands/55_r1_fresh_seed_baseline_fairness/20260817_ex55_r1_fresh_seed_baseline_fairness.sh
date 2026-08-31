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
if [[ ! "${STAGE}" =~ ^(preflight|pilot|formal|verify|resume|all)$ ]]; then
  echo "Usage: bash commands/55_r1_fresh_seed_baseline_fairness/20260817_ex55_r1_fresh_seed_baseline_fairness.sh {preflight|pilot|formal|verify|resume|all}" >&2
  exit 2
fi

EX55_REPO="${EX55_REPO:-${SNM_REPO}}"
EX55_WORKSPACE="${EX55_WORKSPACE:-${SNM_ARTIFACT_ROOT}}"
EX55_CONTROLLER_PYTHON="${EX55_CONTROLLER_PYTHON:-${SNM_CONTROLLER_PYTHON}}"
EX55_TRAINING_PYTHON="${EX55_TRAINING_PYTHON:-${SNM_TRAINING_PYTHON}}"
EX55_OFFICIAL_REPO="${EX55_OFFICIAL_REPO:-${SNM_OFFICIAL_REPO}}"
EX55_RESULT_ROOT="${EX55_RESULT_ROOT:-${SNM_RESULTS_ROOT}/55_r1_fresh_seed_baseline_fairness}"
EX55_GPUS="${EX55_GPUS:-1}"
EX55_WANDB_MODE="${EX55_WANDB_MODE:-disabled}"
EX55_WANDB_PROJECT="${EX55_WANDB_PROJECT:-anonymous-optimizer-artifact-ex55}"
EX55_RUN_DIR="${EX55_RUN_DIR:-${RUN_DIR:-${EX55_RESULT_ROOT}/$(date -u +%Y%m%dT%H%M%S+0000)}}"

EX55_ENCODED_INPUT_ROOT="${EX55_REPO}/scripts/55_r1_fresh_seed_baseline_fairness/accepted_inputs_encoded"

read -r -a GPU_ARGS <<< "${EX55_GPUS}"
if [[ "${#GPU_ARGS[@]}" -ne 1 ]] || [[ "${GPU_ARGS[0]}" != "1" ]]; then
  echo "EX55 is frozen to EX55_GPUS='1'; observed '${EX55_GPUS}'" >&2
  exit 2
fi
for executable in "${EX55_CONTROLLER_PYTHON}" "${EX55_TRAINING_PYTHON}"; do
  if [[ ! -x "${executable}" ]]; then
    echo "Python is not executable: ${executable}" >&2
    exit 2
  fi
done
if [[ ! -f "${EX55_OFFICIAL_REPO}/train_gpt_newton_muon_1.py" ]]; then
  echo "Official R0 repository is incomplete: ${EX55_OFFICIAL_REPO}" >&2
  exit 2
fi

EX55_DEFAULT_INPUT_ROOT="${EX55_RUN_DIR}/accepted_inputs"
for local_name in historical_panel.csv extended_selection.csv mousse_selection.json malt_selection.json; do
  if [[ ! -f "${EX55_DEFAULT_INPUT_ROOT}/${local_name}" ]]; then
    EX55_DEFAULT_INPUT_ROOT="${EX55_RUN_DIR}/materialized_accepted_inputs"
    "${EX55_CONTROLLER_PYTHON}" -B \
      "${EX55_REPO}/scripts/55_r1_fresh_seed_baseline_fairness/materialize_accepted_inputs.py" \
      --encoded-root "${EX55_ENCODED_INPUT_ROOT}" \
      --output-root "${EX55_DEFAULT_INPUT_ROOT}"
    break
  fi
done
EX55_HISTORICAL_PANEL="${EX55_HISTORICAL_PANEL:-${EX55_DEFAULT_INPUT_ROOT}/historical_panel.csv}"
EX55_EXTENDED_SELECTION="${EX55_EXTENDED_SELECTION:-${EX55_DEFAULT_INPUT_ROOT}/extended_selection.csv}"
EX55_MOUSSE_SELECTION="${EX55_MOUSSE_SELECTION:-${EX55_DEFAULT_INPUT_ROOT}/mousse_selection.json}"
EX55_MALT_SELECTION="${EX55_MALT_SELECTION:-${EX55_DEFAULT_INPUT_ROOT}/malt_selection.json}"

declare -a ACCEPTED_INPUT_SPECS=(
  "historical_panel.csv|${EX55_HISTORICAL_PANEL}"
  "extended_selection.csv|${EX55_EXTENDED_SELECTION}"
  "mousse_selection.json|${EX55_MOUSSE_SELECTION}"
  "malt_selection.json|${EX55_MALT_SELECTION}"
)
for spec in "${ACCEPTED_INPUT_SPECS[@]}"; do
  local_name="${spec%%|*}"
  source_path="${spec#*|}"
  run_copy="${EX55_RUN_DIR}/accepted_inputs/${local_name}"
  if [[ ! -f "${run_copy}" ]] && [[ ! -f "${source_path}" ]]; then
    echo "Required accepted EX55 input is absent both at source and in-run copy: ${source_path}" >&2
    exit 2
  fi
done

if [[ "${STAGE}" == "resume" ]]; then
  if [[ ! -d "${EX55_RUN_DIR}" ]] || [[ -z "$(find "${EX55_RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "EX55 resume requires an existing nonempty EX55_RUN_DIR: ${EX55_RUN_DIR}" >&2
    exit 2
  fi
  CONTROLLER_STAGE="all"
else
  CONTROLLER_STAGE="${STAGE}"
fi

SUITE="${EX55_REPO}/scripts/55_r1_fresh_seed_baseline_fairness/run_fresh_seed_suite.py"
SNAPSHOT_SUITE="${EX55_RUN_DIR}/source_snapshot/scripts/55_r1_fresh_seed_baseline_fairness/run_fresh_seed_suite.py"
# Existing runs created before either compatibility fix use the live amended
# controller and record amendment hashes in-run. Fresh snapshots contain both
# the formal-smoke initial_validation_evidence repair and the formal-metrics
# tail-5 lineage marker, and therefore remain fully snapshot-driven.
if [[ -f "${SNAPSHOT_SUITE}" ]] \
  && grep -q 'initial_validation_evidence' "${SNAPSHOT_SUITE}" \
  && grep -q 'FORMAL_METRICS_TAIL5_LINEAGE = True' "${SNAPSHOT_SUITE}"; then
  SUITE="${SNAPSHOT_SUITE}"
fi

echo "EX55_RUN_DIR=${EX55_RUN_DIR}"
echo "EX55_CONTROLLER_PYTHON=${EX55_CONTROLLER_PYTHON}"
echo "EX55_TRAINING_PYTHON=${EX55_TRAINING_PYTHON}"
echo "EX55_OFFICIAL_REPO=${EX55_OFFICIAL_REPO}"
echo "EX55_GPUS=${EX55_GPUS}"
echo "EX55_WANDB_MODE=${EX55_WANDB_MODE}"

if [[ "${STAGE}" == "preflight" ]] || [[ "${STAGE}" == "all" ]]; then
  "${EX55_CONTROLLER_PYTHON}" -B "${EX55_REPO}/scripts/55_r1_fresh_seed_baseline_fairness/test_analyze_fresh_seed_panel.py"
  "${EX55_CONTROLLER_PYTHON}" -B "${EX55_REPO}/scripts/55_r1_fresh_seed_baseline_fairness/test_fresh_seed_suite.py"
fi

RESUME_ARGS=()
if [[ -d "${EX55_RUN_DIR}" ]] && [[ -n "$(find "${EX55_RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  RESUME_ARGS+=(--resume)
fi

COMMAND=(
  "${EX55_CONTROLLER_PYTHON}" -B "${SUITE}"
  --stage "${CONTROLLER_STAGE}"
  --run-dir "${EX55_RUN_DIR}"
  --repo "${EX55_REPO}"
  --official-repo "${EX55_OFFICIAL_REPO}"
  --training-python "${EX55_TRAINING_PYTHON}"
  --gpus "${GPU_ARGS[@]}"
  --historical-panel "${EX55_HISTORICAL_PANEL}"
  --extended-selection "${EX55_EXTENDED_SELECTION}"
  --mousse-selection "${EX55_MOUSSE_SELECTION}"
  --malt-selection "${EX55_MALT_SELECTION}"
  --wandb-mode "${EX55_WANDB_MODE}"
  --wandb-project "${EX55_WANDB_PROJECT}"
  "${RESUME_ARGS[@]}"
)
if [[ -n "${EX55_WANDB_ENTITY:-}" ]]; then
  COMMAND+=(--wandb-entity "${EX55_WANDB_ENTITY}")
fi
"${COMMAND[@]}"

echo "EX55_ARTIFACTS=${EX55_RUN_DIR}"
