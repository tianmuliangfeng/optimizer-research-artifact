#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORKDIR="${SNM_REPO:-${ARTIFACT_ROOT}}"

OFFICIAL_REPO="${OFFICIAL_REPO:-${SNM_OFFICIAL_REPO:-${ARTIFACT_ROOT}/third_party/Newton-Muon-official-r0}}"
DEFAULT_PYTHON="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
CTRL_PY="${CTRL_PY:-${SNM_CONTROLLER_PYTHON:-${DEFAULT_PYTHON:-python3}}}"
TRAIN_PY="${TRAIN_PY:-${SNM_TRAINING_PYTHON:-${DEFAULT_PYTHON:-python3}}}"
LR_CROSS_GPU="${R1_LR_CROSS_GPU:-1}"
RUNNER="$SCRIPT_DIR/run_official_newton_muon_r1_lr_cross.py"
RESULTS_BASE="${SNM_RESULTS_ROOT:-${ARTIFACT_ROOT}/runs}"
RESULTS_ROOT="${R1_LR_CROSS_RESULTS_ROOT:-${RESULTS_BASE}/16_official_newton_muon_r1_lr_cross/results}"
LAUNCH_LOG_DIR="$RESULTS_ROOT/launcher_logs"

for required_file in "$CTRL_PY" "$TRAIN_PY" "$RUNNER"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: required file is missing: $required_file" >&2
        exit 1
    fi
done
if [[ ! -d "$OFFICIAL_REPO" ]]; then
    echo "ERROR: official repository is missing: $OFFICIAL_REPO" >&2
    exit 1
fi

mkdir -p "$LAUNCH_LOG_DIR"
cd "$WORKDIR"

latest_batch() {
    local seed_results="$1"
    local kind="$2"
    local seed="$3"
    find "$seed_results" -mindepth 1 -maxdepth 1 -type d \
        -name "*_${kind}_seed${seed}" -print 2>/dev/null \
        | sort \
        | tail -n 1
}

run_seed() {
    local seed="$1"
    shift
    local -a methods=("$@")
    local seed_results="$RESULTS_ROOT/seed${seed}"
    local tag
    local smoke_log
    local formal_log
    local smoke_dir
    local formal_dir

    mkdir -p "$seed_results"
    tag="$(date -u +%Y%m%dT%H%M%SZ)"
    smoke_log="$LAUNCH_LOG_DIR/seed${seed}_${tag}_smoke.log"
    formal_log="$LAUNCH_LOG_DIR/seed${seed}_${tag}_formal.log"

    echo "===== LR-cross seed ${seed}: methods ${methods[*]} ====="
    smoke_dir="$(latest_batch "$seed_results" smoke "$seed")"
    if [[ -n "$smoke_dir" ]]; then
        echo "Resume/revalidate smoke batch: $smoke_dir"
        CUDA_VISIBLE_DEVICES="$LR_CROSS_GPU" "$CTRL_PY" "$RUNNER" \
            --official-repo "$OFFICIAL_REPO" \
            --python-exe "$TRAIN_PY" \
            --seed "$seed" \
            --methods "${methods[@]}" \
            --numerical-smoke \
            --smoke-steps 10 \
            --resume-batch "$smoke_dir" \
            --results-dir "$seed_results" \
            --wandb-mode disabled \
            2>&1 | tee "$smoke_log"
    else
        echo "Create exact-shape smoke batch"
        CUDA_VISIBLE_DEVICES="$LR_CROSS_GPU" "$CTRL_PY" "$RUNNER" \
            --official-repo "$OFFICIAL_REPO" \
            --python-exe "$TRAIN_PY" \
            --seed "$seed" \
            --methods "${methods[@]}" \
            --numerical-smoke \
            --smoke-steps 10 \
            --results-dir "$seed_results" \
            --wandb-mode disabled \
            2>&1 | tee "$smoke_log"
        smoke_dir="$(sed -n 's/^R1 artifacts: //p' "$smoke_log" | tail -n 1)"
    fi

    if [[ -z "$smoke_dir" || ! -f "$smoke_dir/r1_manifest.json" ]]; then
        echo "ERROR: valid smoke artifact was not found for seed ${seed}" >&2
        exit 1
    fi

    formal_dir="$(latest_batch "$seed_results" formal "$seed")"
    if [[ -n "$formal_dir" ]]; then
        echo "Resume/revalidate formal batch: $formal_dir"
        CUDA_VISIBLE_DEVICES="$LR_CROSS_GPU" "$CTRL_PY" "$RUNNER" \
            --official-repo "$OFFICIAL_REPO" \
            --python-exe "$TRAIN_PY" \
            --seed "$seed" \
            --methods "${methods[@]}" \
            --resume-batch "$formal_dir" \
            --results-dir "$seed_results" \
            --wandb-mode online \
            2>&1 | tee "$formal_log"
    else
        echo "Create formal batch from: $smoke_dir/r1_manifest.json"
        CUDA_VISIBLE_DEVICES="$LR_CROSS_GPU" "$CTRL_PY" "$RUNNER" \
            --official-repo "$OFFICIAL_REPO" \
            --python-exe "$TRAIN_PY" \
            --seed "$seed" \
            --methods "${methods[@]}" \
            --smoke-manifest "$smoke_dir/r1_manifest.json" \
            --results-dir "$seed_results" \
            --wandb-mode online \
            2>&1 | tee "$formal_log"
    fi
    echo "===== LR-cross seed ${seed} complete ====="
}

# Reverse the corresponding formal-R1 endpoint order within each seed. This
# counterbalances method order without changing any model/data configuration.
run_seed 2024 muon diag
run_seed 2025 diag muon
run_seed 2026 muon diag

echo "===== all R1 LR-cross seeds completed ====="
