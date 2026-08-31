#!/usr/bin/env bash
set -Eeuo pipefail

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

MODE="${1:-check}"
if [[ ! "$MODE" =~ ^(check|preflight|tuning|formal|resume|verify|all)$ ]]; then
  echo "usage: $0 {check|preflight|tuning|formal|resume|verify|all}" >&2
  exit 2
fi

REPO="${EX57_REPO:-${SNM_REPO}}"
WORKSPACE="${EX57_WORKSPACE:-${SNM_ARTIFACT_ROOT}}"

# EX57 is frozen to these exact runtimes.  Keeping the virtualenv launcher
# path is essential: resolving venv/bin/python to /usr/bin/python3.10 would
# silently switch the nested probe/workers to the system PyTorch stack.
CONTROLLER_PYTHON="${EX57_CONTROLLER_PYTHON:-${SNM_CONTROLLER_PYTHON}}"
TRAINING_PYTHON="${EX57_TRAINING_PYTHON:-${SNM_TRAINING_PYTHON}}"

OFFICIAL_REPO="${EX57_OFFICIAL_REPO:-${SNM_OFFICIAL_REPO}}"
DATA_DIR="${EX57_DATA_DIR:-${OFFICIAL_REPO}/data/fineweb10B}"
RESULT_ROOT="${EX57_RESULT_ROOT:-${SNM_RESULTS_ROOT}/57_llama1b_10b_moonlight}"
ACTIVE_RUN_FILE="${EX57_ACTIVE_RUN_FILE:-${RESULT_ROOT}/LATEST_EX57_MOONLIGHT_RUN.txt}"
PROTOCOL_TAG="moonlight_fair_parallel_ex19_exact_v4"
PROTOCOL_MARKER_NAME=".ex57_launcher_protocol_v4"
WANDB_MODE="${EX57_WANDB_MODE:-disabled}"
WANDB_PROJECT="${EX57_WANDB_PROJECT:-anonymous-optimizer-artifact-ex57}"

read -r -a GPUS <<< "${EX57_GPUS:-0 1 2}"
if [[ "${#GPUS[@]}" -ne 3 || "${GPUS[0]}" != "0" || "${GPUS[1]}" != "1" || "${GPUS[2]}" != "2" ]]; then
  echo "EX57 Moonlight is frozen to physical GPUs 0 1 2" >&2
  exit 2
fi

for required in "$CONTROLLER_PYTHON" "$TRAINING_PYTHON"; do
  if [[ ! -x "$required" ]]; then
    echo "Missing executable: $required" >&2
    exit 2
  fi
done

CONTROLLER="${REPO}/scripts/57_llama1b_10b_moonlight/run_suite.py"
if [[ ! -f "$CONTROLLER" ]]; then
  echo "Missing EX57 controller: $CONTROLLER" >&2
  exit 2
fi

utc_stamp() {
  date -u +%Y%m%dT%H%M%S+0000
}

absolute_path_without_resolving_symlinks() {
  local value="$1"
  if [[ "$value" == /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s/%s\n' "$PWD" "$value"
  fi
}

new_run_dir() {
  local base candidate suffix
  base="${RESULT_ROOT}/$(utc_stamp)"
  candidate="$base"
  suffix=1
  while [[ -e "$candidate" ]]; do
    printf -v candidate '%s_%02d' "$base" "$suffix"
    suffix=$((suffix + 1))
  done
  printf '%s\n' "$candidate"
}

write_active_run() {
  local run_dir="$1"
  local tmp
  mkdir -p "$RESULT_ROOT"
  tmp="${ACTIVE_RUN_FILE}.tmp.$$"
  printf '%s\n' "$run_dir" > "$tmp"
  mv -f "$tmp" "$ACTIVE_RUN_FILE"
}

mark_protocol_run() {
  local run_dir="$1"
  printf '%s\n' "$PROTOCOL_TAG" > "$run_dir/$PROTOCOL_MARKER_NAME"
}

require_protocol_run() {
  local run_dir="$1"
  local marker="$run_dir/$PROTOCOL_MARKER_NAME"
  if [[ ! -s "$marker" ]] || [[ "$(cat "$marker")" != "$PROTOCOL_TAG" ]]; then
    echo "EX57 resume target belongs to an older Moonlight protocol: $run_dir" >&2
    echo "Start the corrected GPU-parallel protocol with: bash $0 all" >&2
    return 1
  fi
}

read_active_run() {
  local candidate=""
  if [[ -s "$ACTIVE_RUN_FILE" ]]; then
    IFS= read -r candidate < "$ACTIVE_RUN_FILE" || true
    if [[ -n "$candidate" && -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi
  return 1
}

find_latest_existing_run() {
  local row candidate=""
  [[ -d "$RESULT_ROOT" ]] || return 1
  row="$(
    find "$RESULT_ROOT" -mindepth 1 -maxdepth 1 -type d \
      -printf '%T@\t%p\n' 2>/dev/null \
      | sort -nr \
      | head -n 1 \
      || true
  )"
  [[ -n "$row" ]] || return 1
  candidate="${row#*$'\t'}"
  [[ -d "$candidate" ]] || return 1
  printf '%s\n' "$candidate"
}

select_existing_run() {
  local candidate=""

  # Ordinary resume is pointer-driven and immune to stale EX57_RUN_DIR values
  # from retired Mousse / earlier Moonlight launchers. For deliberate manual
  # recovery, use EX57_RESUME_RUN_DIR=/exact/run/path.
  if [[ -n "${EX57_RESUME_RUN_DIR:-}" ]]; then
    candidate="$(absolute_path_without_resolving_symlinks "$EX57_RESUME_RUN_DIR")"
  elif candidate="$(read_active_run)"; then
    :
  elif candidate="$(find_latest_existing_run)"; then
    echo "EX57 active-run pointer was missing; selected latest run: $candidate" >&2
  else
    echo "No existing EX57 Moonlight run was found under: $RESULT_ROOT" >&2
    echo "Start a new run first with: bash $0 all" >&2
    return 1
  fi

  if [[ ! -d "$candidate" ]]; then
    echo "EX57 resume target does not exist: $candidate" >&2
    return 1
  fi
  require_protocol_run "$candidate"
  printf '%s\n' "$candidate"
}

# Run-directory policy:
#   all       -> create and remember a fresh independent EX57 run;
#   resume    -> reopen the remembered run (or latest existing run as fallback);
#   preflight -> create and remember a fresh run when no explicit run is given;
#   tuning/formal/verify -> continue the remembered run;
#   check     -> does not create or change the remembered run.
case "$MODE" in
  check)
    RUN_DIR="${EX57_RUN_DIR:-${RESULT_ROOT}/_check_only}"
    ;;
  all)
    # A new scientific protocol (v4 exact EX19 transfer + fairness + three-GPU tuning) must always
    # start a clean lineage. Ignore stale EX57_RUN_DIR exports.
    if [[ -n "${EX57_RUN_DIR:-}" ]]; then
      echo "EX57_NOTICE=ignoring legacy EX57_RUN_DIR during all: ${EX57_RUN_DIR}" >&2
    fi
    RUN_DIR="$(new_run_dir)"
    mkdir -p "$RUN_DIR"
    mark_protocol_run "$RUN_DIR"
    write_active_run "$RUN_DIR"
    ;;
  preflight)
    RUN_DIR="$(new_run_dir)"
    mkdir -p "$RUN_DIR"
    mark_protocol_run "$RUN_DIR"
    write_active_run "$RUN_DIR"
    ;;
  tuning|formal|verify|resume)
    RUN_DIR="$(select_existing_run)"
    write_active_run "$RUN_DIR"
    ;;
esac

ARGS=(
  "$CONTROLLER_PYTHON" -B "$CONTROLLER" "$MODE"
  --run-dir "$RUN_DIR" --repo "$REPO" --official-repo "$OFFICIAL_REPO"
  --data-dir "$DATA_DIR" --training-python "$TRAINING_PYTHON"
  --gpus "${GPUS[@]}" --wandb-mode "$WANDB_MODE" --wandb-project "$WANDB_PROJECT"
)
if [[ -n "${EX57_WANDB_ENTITY:-}" ]]; then
  ARGS+=(--wandb-entity "$EX57_WANDB_ENTITY")
fi

if [[ "$MODE" == "all" || "$MODE" == "resume" ]]; then
  LOG_FILE="${RUN_DIR}/full_pipeline.log"
elif [[ "$MODE" == "check" ]]; then
  LOG_FILE=""
else
  LOG_FILE="${RUN_DIR}/${MODE}.log"
fi

print_header() {
  echo "EX57_RUN_DIR=$RUN_DIR"
  echo "EX57_ACTIVE_RUN_FILE=$ACTIVE_RUN_FILE"
  echo "EX57_GPUS=${GPUS[*]}"
  echo "EX57_METHOD=moonlight"
  echo "EX57_PROTOCOL=$PROTOCOL_TAG"
  echo "EX57_FAIRNESS=device_batch_8_accumulation_64_global_batch_512"
  echo "EX57_MOONLIGHT_LINEAGE=EX19_exact_algorithm_subtrees backup=accepted_llama_adamw"
  echo "EX57_COMPILE_CACHE=per_physical_gpu"
  echo "EX57_TUNING_SEED=5701 formal_seeds=2024,2025,2026 overlap=false"
  echo "EX57_PARALLEL=tuning=3xsingleGPU formal=3xsingleGPU ddp=false"
  echo "EX57_CONTROLLER_PYTHON=$CONTROLLER_PYTHON"
  echo "EX57_TRAINING_PYTHON=$TRAINING_PYTHON"
  if [[ -n "$LOG_FILE" ]]; then
    echo "EX57_LOG=$LOG_FILE"
  fi
}

run_runtime_banner() {
  "$TRAINING_PYTHON" - <<'PY'
import sys
import numpy
import torch
import triton
print(
    "EX57_RUNTIME="
    f"executable={sys.executable} "
    f"prefix={sys.prefix} "
    f"python={sys.version.split()[0]} "
    f"torch={torch.__version__} "
    f"torch_cuda={torch.version.cuda} "
    f"triton={triton.__version__} "
    f"numpy={numpy.__version__} "
    f"torch_file={torch.__file__}"
)
PY
}

run_controller() {
  # Explicit command-line paths are the sole runtime source of truth.  Remove
  # stale interpreter overrides inherited from previous EX57 attempts.
  env -u EX57_CONTROLLER_PYTHON -u EX57_TRAINING_PYTHON \
    PYTHONDONTWRITEBYTECODE=1 "${ARGS[@]}"
}

print_header

set +e
if [[ "$MODE" == "check" ]]; then
  (
    set -Eeuo pipefail
    run_runtime_banner
    run_controller
  )
  RC=$?
else
  (
    set -Eeuo pipefail
    echo "EX57_STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    run_runtime_banner
    run_controller
    echo "EX57_FINISHED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  ) 2>&1 | tee -a "$LOG_FILE"
  RC=${PIPESTATUS[0]}
fi
set -e

if [[ "$RC" -ne 0 ]]; then
  echo >&2
  echo "EX57 Moonlight stopped with rc=$RC" >&2
  if [[ "$MODE" != "check" ]]; then
    echo "EX57_RUN_DIR=$RUN_DIR" >&2
    echo "Resume with:" >&2
    echo "  bash commands/57_llama1b_10b_moonlight/20260819_ex57_llama1b_10b_moonlight.sh resume" >&2
  fi
  exit "$RC"
fi

echo "EX57_ARTIFACTS=$RUN_DIR"
if [[ "$MODE" == "all" || "$MODE" == "resume" ]]; then
  echo "EX57_SHORT_RESUME=bash commands/57_llama1b_10b_moonlight/20260819_ex57_llama1b_10b_moonlight.sh resume"
fi
