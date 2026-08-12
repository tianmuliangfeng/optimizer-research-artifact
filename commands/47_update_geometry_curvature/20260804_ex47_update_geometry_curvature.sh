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

# Experiment 47 / GEO-01 outcome-blind check, sealed dry-run, H100 engineering
# pilot, same-contract resume, and read-only verification.

MODE="${1:-check}"
REPO="${REPO:-${SNM_REPO}}"
CONTROLLER_PYTHON="${GEO01_CONTROLLER_PYTHON:-${SNM_CONTROLLER_PYTHON}}"
SCRIPT="$REPO/scripts/47_update_geometry_curvature/run_geo01.py"
CONTROLLER="$REPO/scripts/47_update_geometry_curvature/remote_controller.py"
SOURCE_RUN="${SOURCE_RUN:-${SNM_RESULTS_ROOT}/37_mech09_downproj_refresh_mediation/20260728T075907+0000}"
TRAINING_PYTHON="${GEO01_TRAINING_PYTHON:-${SNM_TRAINING_PYTHON}}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${SNM_RESULTS_ROOT}/_shared/analysis/method_deepening_geo01_update_curvature}"
GPUS=(0 1)

echo "GEO01_CONTROLLER_PYTHON=$CONTROLLER_PYTHON"
echo "GEO01_TRAINING_PYTHON=$TRAINING_PYTHON"

case "$MODE" in
  check)
    env PYTHONDONTWRITEBYTECODE=1 "$CONTROLLER_PYTHON" "$SCRIPT" check
    ;;
  dry-run)
    RUN_DIR="${RUN_DIR:-$ARTIFACT_ROOT/$(date -u +%Y%m%dT%H%M%S+0000)_dryrun}"
    env PYTHONDONTWRITEBYTECODE=1 "$CONTROLLER_PYTHON" "$CONTROLLER" dry-run \
      --run-dir "$RUN_DIR" --source-run "$SOURCE_RUN" \
      --child-python "$TRAINING_PYTHON" --gpus "${GPUS[@]}"
    echo "GEO01_DRYRUN=$RUN_DIR"
    ;;
  pilot)
    RUN_DIR="${RUN_DIR:-$ARTIFACT_ROOT/$(date -u +%Y%m%dT%H%M%S+0000)}"
    env PYTHONDONTWRITEBYTECODE=1 "$CONTROLLER_PYTHON" "$CONTROLLER" pilot \
      --run-dir "$RUN_DIR" --source-run "$SOURCE_RUN" \
      --child-python "$TRAINING_PYTHON" --gpus "${GPUS[@]}"
    echo "GEO01_ARTIFACTS=$RUN_DIR"
    ;;
  resume)
    RUN_DIR="${RUN_DIR:?set RUN_DIR to the existing GEO-01 pilot directory}"
    env PYTHONDONTWRITEBYTECODE=1 "$CONTROLLER_PYTHON" "$CONTROLLER" resume \
      --run-dir "$RUN_DIR" --source-run "$SOURCE_RUN" \
      --child-python "$TRAINING_PYTHON" --gpus "${GPUS[@]}"
    ;;
  verify)
    RUN_DIR="${RUN_DIR:?set RUN_DIR to the completed GEO-01 pilot directory}"
    env PYTHONDONTWRITEBYTECODE=1 "$CONTROLLER_PYTHON" "$CONTROLLER" verify --run-dir "$RUN_DIR"
    ;;
  discovery|confirmation|llama-10b)
    echo "GEO-01 $MODE remains gated and is not authorized by the pilot." >&2
    exit 2
    ;;
  *)
    echo "usage: $0 {check|dry-run|pilot|resume|verify}" >&2
    exit 2
    ;;
esac
