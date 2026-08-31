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
if [[ ! "$MODE" =~ ^(check|preflight|tuning|formal_non10b|formal|resume|verify|upload|all)$ ]]; then
  echo "usage: $0 {check|preflight|tuning|formal_non10b|formal|resume|verify|upload|all}" >&2
  exit 2
fi

REPO="${EX54_REPO:-${SNM_REPO}}"
WORKSPACE="${EX54_WORKSPACE:-${SNM_ARTIFACT_ROOT}}"
OFFICIAL_REPO="${EX54_OFFICIAL_REPO:-${SNM_OFFICIAL_REPO}}"

# Keep the virtual-environment launcher paths intact. Resolving these symlinks
# to /usr/bin/python can silently switch nested probes/workers to the system
# PyTorch stack instead of the frozen torch-2.8/cu126 environment.
CONTROLLER_PYTHON="${EX54_CONTROLLER_PYTHON:-${SNM_CONTROLLER_PYTHON}}"
TRAINING_PYTHON="${EX54_TRAINING_PYTHON:-${SNM_TRAINING_PYTHON}}"

DATA1B_DIR="${EX54_DATA1B_DIR:-${OFFICIAL_REPO}/data/fineweb10B}"
DATA124_SOURCE_DIR="${EX54_DATA124_SOURCE_DIR:-${DATA1B_DIR}}"
# Do not inherit the legacy EX54_DATA124_DIR knob: an old value commonly points
# at the 103-shard 1B directory and recreates the exact preflight failure this
# launcher is repairing.  Use EX54_DATA124_VIEW_DIR only for an intentional
# alternate location of the same frozen 50-shard symlink view.
if [[ -n "${EX54_DATA124_DIR:-}" ]]; then
  echo "EX54_NOTICE=ignoring legacy EX54_DATA124_DIR=${EX54_DATA124_DIR}" >&2
fi
DATA124_DIR="${EX54_DATA124_VIEW_DIR:-${SNM_RESULTS_ROOT}/_data_views/ex54_ex52_frozen50/fineweb10B}"

RESULT_ROOT="${EX54_RESULT_ROOT:-${SNM_RESULTS_ROOT}/54_llama_moonlight_multiscale_multibudget}"
ACTIVE_RUN_FILE="${EX54_ACTIVE_RUN_FILE:-${RESULT_ROOT}/LATEST_EX54_MOONLIGHT_RUN.txt}"
PROTOCOL_TAG="moonlight_fair_parallel_ex19_exact_v4"
PROTOCOL_MARKER_NAME=".ex54_launcher_protocol_v4"
WANDB_MODE="${EX54_WANDB_MODE:-disabled}"
WANDB_PROJECT="${EX54_WANDB_PROJECT:-anonymous-optimizer-artifact-ex54}"

CONTROLLER="${REPO}/scripts/54_llama_moonlight_multiscale_multibudget/run_suite.py"
VIEW_BUILDER="${REPO}/scripts/54_llama_moonlight_multiscale_multibudget/prepare_frozen50_view.py"
PROJECTION="${REPO}/scripts/54_llama_moonlight_multiscale_multibudget/accepted_ex48_data_projection.json"
CONTRACT="${REPO}/scripts/54_llama_moonlight_multiscale_multibudget/ex54_contract.json"

read -r -a GPUS <<< "${EX54_GPUS:-0 1}"
if [[ "${#GPUS[@]}" -ne 2 || "${GPUS[0]}" != "0" || "${GPUS[1]}" != "1" ]]; then
  echo "EX54 Moonlight is frozen to physical GPUs 0 1" >&2
  exit 2
fi

for required in "$CONTROLLER_PYTHON" "$TRAINING_PYTHON"; do
  if [[ ! -x "$required" ]]; then
    echo "Missing executable: $required" >&2
    exit 2
  fi
done
for required in "$CONTROLLER" "$VIEW_BUILDER" "$PROJECTION" "$CONTRACT"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing EX54 file: $required" >&2
    exit 2
  fi
done

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
    echo "EX54 resume target belongs to an older Moonlight protocol: $run_dir" >&2
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
    find "$RESULT_ROOT" -mindepth 1 -maxdepth 1 -type d -name '20*' \
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
  # Ordinary resume is pointer-driven and is immune to stale EX54_RUN_DIR
  # exports left by the retired Mousse/old-Moonlight launchers.  A deliberate
  # manual recovery may opt in with EX54_RESUME_RUN_DIR=/exact/run/path.
  if [[ -n "${EX54_RESUME_RUN_DIR:-}" ]]; then
    candidate="$(absolute_path_without_resolving_symlinks "$EX54_RESUME_RUN_DIR")"
  elif candidate="$(read_active_run)"; then
    :
  elif candidate="$(find_latest_existing_run)"; then
    echo "EX54 active-run pointer was missing; selected latest run: $candidate" >&2
  else
    echo "No existing EX54 Moonlight run was found under: $RESULT_ROOT" >&2
    echo "Start a new run first with: bash $0 all" >&2
    return 1
  fi
  if [[ ! -d "$candidate" ]]; then
    echo "EX54 resume target does not exist: $candidate" >&2
    return 1
  fi
  require_protocol_run "$candidate"
  printf '%s\n' "$candidate"
}

prepare_data124_view() {
  if [[ ! -d "$DATA124_SOURCE_DIR" ]]; then
    echo "EX54 full FineWeb source directory is absent: $DATA124_SOURCE_DIR" >&2
    exit 2
  fi
  "$CONTROLLER_PYTHON" -B "$VIEW_BUILDER" \
    --source-dir "$DATA124_SOURCE_DIR" \
    --view-dir "$DATA124_DIR" \
    --projection "$PROJECTION" \
    --contract "$CONTRACT"

  local train_count val_count
  train_count="$(
    find -L "$DATA124_DIR" -maxdepth 1 -type f \
      -name 'fineweb_train_*.bin' -printf '.\n' 2>/dev/null \
      | wc -l
  )"
  val_count="$(
    find -L "$DATA124_DIR" -maxdepth 1 -type f \
      -name 'fineweb_val_*.bin' -printf '.\n' 2>/dev/null \
      | wc -l
  )"
  if [[ "$train_count" -ne 50 || "$val_count" -ne 1 ]]; then
    echo "EX54 124M data view must contain exactly 50 train shards and 1 validation shard." >&2
    echo "Observed: train=$train_count validation=$val_count path=$DATA124_DIR" >&2
    echo "Do not point EX54_DATA124_DIR at the 103-shard fineweb10B directory." >&2
    exit 2
  fi
}

# Run-directory policy:
#   all       -> create and remember a fresh, independent EX54 run;
#   resume    -> reopen the remembered run (latest existing run as fallback);
#   preflight -> create and remember a fresh run unless explicitly overridden;
#   later stages continue the remembered run;
#   check     -> read-only and does not change the active-run pointer.
case "$MODE" in
  check)
    RUN_DIR="${RESULT_ROOT}/_check_only"
    ;;
  all)
    # `all` is always a scientifically clean, fresh run.  Ignore stale
    # EX54_RUN_DIR exports from older launchers.
    if [[ -n "${EX54_RUN_DIR:-}" ]]; then
      echo "EX54_NOTICE=ignoring legacy EX54_RUN_DIR during all: ${EX54_RUN_DIR}" >&2
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
  tuning|formal_non10b|formal|verify|upload|resume)
    RUN_DIR="$(select_existing_run)"
    write_active_run "$RUN_DIR"
    ;;
esac

if [[ "$MODE" != "check" && "$MODE" != "upload" ]]; then
  prepare_data124_view
fi

ARGS=(
  "$CONTROLLER_PYTHON" -B "$CONTROLLER" "$MODE"
  --run-dir "$RUN_DIR"
  --repo "$REPO"
  --official-repo "$OFFICIAL_REPO"
  --data124-dir "$DATA124_DIR"
  --data1b-dir "$DATA1B_DIR"
  --training-python "$TRAINING_PYTHON"
  --gpus "${GPUS[@]}"
  --wandb-mode "$WANDB_MODE"
  --wandb-project "$WANDB_PROJECT"
)
if [[ -n "${EX54_WANDB_ENTITY:-}" ]]; then
  ARGS+=(--wandb-entity "$EX54_WANDB_ENTITY")
fi

if [[ "$MODE" == "all" || "$MODE" == "resume" ]]; then
  LOG_FILE="${RUN_DIR}/full_pipeline.log"
elif [[ "$MODE" == "check" ]]; then
  LOG_FILE=""
else
  LOG_FILE="${RUN_DIR}/${MODE}.log"
fi

print_header() {
  echo "EX54_RUN_DIR=$RUN_DIR"
  echo "EX54_ACTIVE_RUN_FILE=$ACTIVE_RUN_FILE"
  echo "EX54_GPUS=${GPUS[*]}"
  echo "EX54_METHOD=moonlight"
  echo "EX54_PROTOCOL=$PROTOCOL_TAG"
  echo "EX54_FAIRNESS=device_batch_8_accumulation_64_global_batch_512"
  echo "EX54_MOONLIGHT_LINEAGE=EX19_exact_algorithm_subtrees backup=accepted_llama_adamw"
  echo "EX54_COMPILE_CACHE=per_physical_gpu"
  echo "EX54_TUNING_SEEDS=124m:5401,1b:5402 formal_seeds=2024,2025,2026 overlap=false"
  echo "EX54_PARALLEL=tuning=2xsingleGPU formal=2xsingleGPU ddp=false"
  echo "EX54_DATA124_DIR=$DATA124_DIR"
  echo "EX54_DATA1B_DIR=$DATA1B_DIR"
  echo "EX54_CONTROLLER_PYTHON=$CONTROLLER_PYTHON"
  echo "EX54_TRAINING_PYTHON=$TRAINING_PYTHON"
  if [[ -n "$LOG_FILE" ]]; then
    echo "EX54_LOG=$LOG_FILE"
  fi
}

run_runtime_banner() {
  "$TRAINING_PYTHON" - <<'PY'
import sys
import numpy
import torch
import triton
print(
    "EX54_RUNTIME="
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
  env -u EX54_CONTROLLER_PYTHON -u EX54_TRAINING_PYTHON \
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
    echo "EX54_STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    run_runtime_banner
    run_controller
    echo "EX54_FINISHED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  ) 2>&1 | tee -a "$LOG_FILE"
  RC=${PIPESTATUS[0]}
fi
set -e

if [[ "$RC" -ne 0 ]]; then
  echo >&2
  echo "EX54 Moonlight stopped with rc=$RC" >&2
  if [[ "$MODE" != "check" ]]; then
    echo "EX54_RUN_DIR=$RUN_DIR" >&2
    echo "Resume with:" >&2
    echo "  bash commands/54_llama_moonlight_multiscale_multibudget/20260819_ex54_llama_moonlight_multiscale_multibudget.sh resume" >&2
  fi
  exit "$RC"
fi

echo "EX54_ARTIFACTS=$RUN_DIR"
if [[ "$MODE" == "all" || "$MODE" == "resume" ]]; then
  echo "EX54_SHORT_RESUME=bash commands/54_llama_moonlight_multiscale_multibudget/20260819_ex54_llama_moonlight_multiscale_multibudget.sh resume"
fi
