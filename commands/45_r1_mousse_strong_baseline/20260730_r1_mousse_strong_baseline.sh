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

# Experiment 45: controlled 124M R1 Mousse baseline on the two-GPU R1 host.
# Run in tmux from the main-conference repository. Timing is diagnostic only.
set +e
set +u
set +o pipefail

ROOT=${SNM_REPO}
CTRL_PY=${SNM_CONTROLLER_PYTHON}
TRAIN_PY=${SNM_TRAINING_PYTHON}
OFFICIAL_REPO=${SNM_OFFICIAL_REPO}
RUNNER="$ROOT/scripts/45_r1_mousse_strong_baseline/run_r1_mousse.py"
RESULTS=${SNM_RESULTS_ROOT}/45_r1_mousse_strong_baseline/results
LOGS="$RESULTS/launcher_logs"
LOCKS=${SNM_LOCK_ROOT}

cd "$ROOT" || exit 1
mkdir -p "$LOGS" "$LOCKS"

PRE_LOG="$LOGS/preflight_$(date +%Y%m%dT%H%M%S).log"
CUDA_VISIBLE_DEVICES=0 "$CTRL_PY" -u "$RUNNER" \
  --official-repo "$OFFICIAL_REPO" --python-exe "$TRAIN_PY" \
  --results-dir "$RESULTS" --seed 2026 --preflight 2>&1 | tee "$PRE_LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "Experiment-45 preflight failed. Inspect $PRE_LOG"
  exit 1
fi

PILOT_LOG="$LOGS/pilot_seed2026_$(date +%Y%m%dT%H%M%S).log"
(
  flock -n 9 || { echo "Physical GPU 0 lock is busy"; exit 73; }
  CUDA_VISIBLE_DEVICES=0 "$CTRL_PY" -u "$RUNNER" \
    --official-repo "$OFFICIAL_REPO" --python-exe "$TRAIN_PY" \
    --results-dir "$RESULTS" --seed 2026 --pilot --pilot-steps 1000 \
    --wandb-mode online
) 9>"$LOCKS/gpu0.lock" 2>&1 | tee "$PILOT_LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "Mousse pilot failed or W&B is incomplete. Inspect $PILOT_LOG"
  exit 1
fi

SELECTION=$(sed -n 's/^R1 Mousse selection certificate: //p' "$PILOT_LOG" | tail -n 1)
if [ ! -s "$SELECTION" ]; then
  echo "Pilot selection certificate was not recovered: $SELECTION"
  exit 1
fi

run_seed() {
  local SEED="$1"
  local GPU_ID="$2"
  local LOCK_FILE="$LOCKS/gpu${GPU_ID}.lock"
  local SMOKE_LOG="$LOGS/formal_smoke_seed${SEED}_$(date +%Y%m%dT%H%M%S).log"
  local FORMAL_LOG="$LOGS/formal_seed${SEED}_$(date +%Y%m%dT%H%M%S).log"
  (
    flock -n 9 || { echo "Physical GPU $GPU_ID lock is busy"; exit 73; }
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$CTRL_PY" -u "$RUNNER" \
      --official-repo "$OFFICIAL_REPO" --python-exe "$TRAIN_PY" \
      --results-dir "$RESULTS" --seed "$SEED" --formal-smoke \
      --selection-certificate "$SELECTION" --wandb-mode disabled 2>&1 | tee "$SMOKE_LOG"
    local RUN_STATUS=${PIPESTATUS[0]}
    if [ "$RUN_STATUS" -ne 0 ]; then exit "$RUN_STATUS"; fi
    local SMOKE_BATCH
    SMOKE_BATCH=$(sed -n 's/^R1 Mousse artifacts: //p' "$SMOKE_LOG" | tail -n 1)
    local SMOKE_MANIFEST="$SMOKE_BATCH/formal_smoke_manifest.json"
    if [ ! -s "$SMOKE_MANIFEST" ]; then echo "Missing $SMOKE_MANIFEST"; exit 74; fi
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$CTRL_PY" -u "$RUNNER" \
      --official-repo "$OFFICIAL_REPO" --python-exe "$TRAIN_PY" \
      --results-dir "$RESULTS" --seed "$SEED" --formal \
      --selection-certificate "$SELECTION" --smoke-manifest "$SMOKE_MANIFEST" \
      --wandb-mode online 2>&1 | tee "$FORMAL_LOG"
    RUN_STATUS=${PIPESTATUS[0]}
    if [ "$RUN_STATUS" -ne 0 ]; then exit "$RUN_STATUS"; fi
    local FORMAL_BATCH
    FORMAL_BATCH=$(sed -n 's/^R1 Mousse artifacts: //p' "$FORMAL_LOG" | tail -n 1)
    if [ ! -s "$FORMAL_BATCH/formal_manifest.json" ]; then exit 75; fi
    printf '%s\n' "$FORMAL_BATCH" > "$RESULTS/formal_batch_seed${SEED}.path"
  ) 9>"$LOCK_FILE"
}

run_seed 2026 0 &
PID_2026=$!
run_seed 2024 1 &
PID_2024=$!
wait "$PID_2026"; STATUS_2026=$?
wait "$PID_2024"; STATUS_2024=$?
if [ "$STATUS_2026" -ne 0 ] || [ "$STATUS_2024" -ne 0 ]; then
  echo "Parallel formal wave failed: seed2026=$STATUS_2026 seed2024=$STATUS_2024"
  exit 1
fi

run_seed 2025 0
STATUS_2025=$?
if [ "$STATUS_2025" -ne 0 ]; then
  echo "Formal seed 2025 failed: $STATUS_2025"
  exit 1
fi

echo "Experiment 45 completed locally and uploaded. Formal batch paths:"
for SEED in 2026 2024 2025; do
  cat "$RESULTS/formal_batch_seed${SEED}.path"
done
