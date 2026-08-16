#!/usr/bin/env bash
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
export SNM_ARTIFACT_ROOT SNM_REPO SNM_RESULTS_ROOT SNM_OFFICIAL_REPO
export SNM_CONTROLLER_PYTHON SNM_TRAINING_PYTHON

set -euo pipefail

STAGE="${1:-}"
if [[ ! "${STAGE}" =~ ^(preflight|pilot|screen|formal|verify|all)$ ]]; then
  echo "Usage: bash commands/52_llama_global_diag_scale/20260814_ex52_llama_global_diag_scale.sh {preflight|pilot|screen|formal|verify|all}" >&2
  exit 2
fi
EX52_REPO="${EX52_REPO:-${SNM_REPO}}"
EX52_CONTROLLER_PYTHON="${EX52_CONTROLLER_PYTHON:-${SNM_CONTROLLER_PYTHON}}"
EX52_TRAINING_PYTHON="${EX52_TRAINING_PYTHON:-${SNM_TRAINING_PYTHON}}"
# The accepted LLaMA execution/runtime source is Newton-Muon-official-r0.
# EX52_DATA_DIR is separate because r0 may also contain the later EX48 shards.
EX52_OFFICIAL_REPO="${EX52_OFFICIAL_REPO:-${SNM_OFFICIAL_REPO}}"
EX52_DATA_DIR="${EX52_DATA_DIR:-${EX52_OFFICIAL_REPO}/data/fineweb10B_ex52_frozen50}"
EX52_RESULT_ROOT="${EX52_RESULT_ROOT:-${SNM_RESULTS_ROOT}/52_llama_global_diag_scale}"
EX52_GPUS="${EX52_GPUS:-0 1}"
EX52_WANDB_MODE="${EX52_WANDB_MODE:-disabled}"
EX52_WANDB_PROJECT="${EX52_WANDB_PROJECT:-anonymous-optimizer-artifact-ex52}"
EX52_RUN_DIR="${EX52_RUN_DIR:-${RUN_DIR:-${EX52_RESULT_ROOT}/$(date -u +%Y%m%dT%H%M%S+0000)}}"
read -r -a GPU_ARGS <<< "${EX52_GPUS}"
if [[ "${#GPU_ARGS[@]}" -ne 2 ]]; then echo "EX52 requires exactly two GPU ids" >&2; exit 2; fi
if [[ ! -x "${EX52_CONTROLLER_PYTHON}" ]] || [[ ! -x "${EX52_TRAINING_PYTHON}" ]]; then echo "EX52 Python environment is incomplete" >&2; exit 2; fi
if [[ "${STAGE}" == "preflight" || "${STAGE}" == "all" ]]; then
  "${EX52_CONTROLLER_PYTHON}" -B \
    "${EX52_REPO}/scripts/52_llama_global_diag_scale/prepare_frozen50_view.py" \
    --official-repo "${EX52_OFFICIAL_REPO}" \
    --source-dir "${EX52_OFFICIAL_REPO}/data/fineweb10B" \
    --view-dir "${EX52_DATA_DIR}"
fi
SUITE="${EX52_REPO}/scripts/52_llama_global_diag_scale/run_llama_global_diag_suite.py"
SNAPSHOT_SUITE="${EX52_RUN_DIR}/source_snapshot/scripts/52_llama_global_diag_scale/run_llama_global_diag_suite.py"
if [[ -f "${SNAPSHOT_SUITE}" ]]; then
  SNAPSHOT_CONTRACT="${EX52_RUN_DIR}/source_snapshot/scripts/52_llama_global_diag_scale/llama_global_diag_contract.json"
  SNAPSHOT_VERSION="$("${EX52_CONTROLLER_PYTHON}" -B -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("contract_version", ""))' "${SNAPSHOT_CONTRACT}")"
  if [[ "${SNAPSHOT_VERSION}" == "2026-08-14.4" ]]; then
    SUITE="${SNAPSHOT_SUITE}"
  elif [[ "${SNAPSHOT_VERSION}" == "2026-08-14.3" ]]; then
    # .3 training artifacts are scientifically valid; only its pilot
    # certificate expected formal validation/checkpoint settings.  Run the
    # .4 certificate-only amendment and freeze it inside this run directory.
    AMENDMENT_SUITE="${EX52_RUN_DIR}/controller_amendments/2026-08-14.4/run_llama_global_diag_suite.py"
    if [[ -f "${AMENDMENT_SUITE}" ]]; then
      SUITE="${AMENDMENT_SUITE}"
    else
      SUITE="${EX52_REPO}/scripts/52_llama_global_diag_scale/run_llama_global_diag_suite.py"
    fi
  else
    echo "EX52 refuses unsupported snapshot contract ${SNAPSHOT_VERSION}; start a new EX52_RUN_DIR" >&2
    exit 2
  fi
fi
echo "EX52_RUN_DIR=${EX52_RUN_DIR}"
echo "EX52_OFFICIAL_REPO=${EX52_OFFICIAL_REPO}"
echo "EX52_DATA_DIR=${EX52_DATA_DIR}"
echo "EX52_GPUS=${EX52_GPUS}"
ARGS=("${EX52_CONTROLLER_PYTHON}" -B "${SUITE}" --stage "${STAGE}" --run-dir "${EX52_RUN_DIR}" --repo "${EX52_REPO}" --official-repo "${EX52_OFFICIAL_REPO}" --data-dir "${EX52_DATA_DIR}" --training-python "${EX52_TRAINING_PYTHON}" --gpus "${GPU_ARGS[@]}" --wandb-mode "${EX52_WANDB_MODE}" --wandb-project "${EX52_WANDB_PROJECT}")
if [[ -d "${EX52_RUN_DIR}" ]]; then ARGS+=(--resume); fi
if [[ -n "${EX52_WANDB_ENTITY:-}" ]]; then ARGS+=(--wandb-entity "${EX52_WANDB_ENTITY}"); fi
"${ARGS[@]}"
echo "EX52_ARTIFACTS=${EX52_RUN_DIR}"
