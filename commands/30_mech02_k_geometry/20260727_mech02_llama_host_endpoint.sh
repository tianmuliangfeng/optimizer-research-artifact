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

PROJECT_ROOT="${PROJECT_ROOT:-${SNM_REPO}}"
CTRL_PY="${CTRL_PY:-${SNM_CONTROLLER_PYTHON}}"
TRAIN_PY="${TRAIN_PY:-${SNM_TRAINING_PYTHON}}"
MECH_GPU="${MECH_GPU:-0}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-60}"
RUNNER="$PROJECT_ROOT/scripts/30_mech02_k_geometry/run_mech02.py"

LLAMA_REPO="${SNM_OFFICIAL_REPO}"
BRIDGE_REPO="${SNM_OFFICIAL_REPO}"
LLAMA_TRITON="$LLAMA_REPO/triton_kernels.py"
LLAMA_DATA="$LLAMA_REPO/data/fineweb10B/fineweb_val_*.bin"
BRIDGE_DATA="$BRIDGE_REPO/data/fineweb10B/fineweb_val_*.bin"

GPT_WORKSPACE="${SNM_RESULTS_ROOT}/21_gpt_r1_host_bridge/results/20260721T072708+0000_formal_seed2026/mainconf_gpt_r1_host_bridge_none_seed2026_20260721T072708+0000/workspace"
GPT_CHECKPOINT="$GPT_WORKSPACE/logs/ba9acc67-9a51-4fea-b1bc-695e11bb9c80/state_step006200.pt"
GPT_SHA256="bb9a7d2dcc191cd72caf01e24b838f99c4c319ca3dfdfff036684b939bacef6c"
GPT_SOURCE="$GPT_WORKSPACE/train_r1_none.py"
GPT_TRITON="$GPT_WORKSPACE/triton_kernels.py"
GPT_MECH01="${SNM_RESULTS_ROOT}/27_mech01_unified_k_diagnostics/llama_host_remaining/20260727T045443+0000/gpt_bridge_seed2026_none6200/numerical_smoke"

L124_RUN="${SNM_RESULTS_ROOT}/17_llama_swiglu_validation/results/20260720T062421+0000_formal_seed2026/02_down_none"
L124_CHECKPOINT="$L124_RUN/checkpoint_latest.pt"
L124_SHA256="f13f8b7bd43de31f1f2190702b4954d707bea5489d728ce673145979881682fd"
L124_SOURCE="$L124_RUN/train_llama_swiglu.py"
L124_MECH01="${SNM_RESULTS_ROOT}/27_mech01_unified_k_diagnostics/llama_host_remaining/20260727T045443+0000/llama124_seed2026_down_none6200/numerical_smoke"

STAMP="$(date -u +%Y%m%dT%H%M%S+0000)"
ROOT="${SNM_RESULTS_ROOT}/30_mech02_k_geometry/llama_host_endpoint/$STAMP"

run_with_heartbeat() {
  "$@" &
  local child_pid=$!
  while kill -0 "$child_pid" 2>/dev/null; do
    sleep "$HEARTBEAT_SECONDS"
    if kill -0 "$child_pid" 2>/dev/null; then
      echo "MECH-02 heartbeat $(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$child_pid gpu=$MECH_GPU"
      nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null || true
    fi
  done
  wait "$child_pid"
}

assert_passed() {
  "$CTRL_PY" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("passed") is True, d' \
    "$1"
}

run_family() {
  local slug="$1"
  local family="$2"
  local method="$3"
  local checkpoint="$4"
  local checkpoint_sha="$5"
  local source="$6"
  local triton="$7"
  local data_pattern="$8"
  local mech01="$9"
  local smoke_layers=()

  if [[ "$family" == "gpt_bridge" ]]; then
    smoke_layers=(0 6 11)
  else
    smoke_layers=(0 6 11)
  fi

  local smoke_out="$ROOT/$slug/smoke"
  local formal_out="$ROOT/$slug/formal"

  run_with_heartbeat \
    "$CTRL_PY" "$RUNNER" \
    --analysis-tier smoke \
    --python-exe "$TRAIN_PY" \
    --family "$family" \
    --method "$method" \
    --checkpoint "$checkpoint" \
    --checkpoint-sha256 "$checkpoint_sha" \
    --hash-checkpoint \
    --source-script "$source" \
    --triton-kernels "$triton" \
    --mech01-smoke-dir "$mech01" \
    --data-pattern "$data_pattern" \
    --layers "${smoke_layers[@]}" \
    --repeats 2 \
    --batches-per-repeat 2 \
    --repeat-offsets 0 4096 8192 12288 \
    --host-id llama-host-h100 \
    --execution-domain "llama-host-$family" \
    --output-dir "$smoke_out" \
    --run-prefix "${slug}_smoke"

  assert_passed "$smoke_out/mech02_manifest.json"

  run_with_heartbeat \
    "$CTRL_PY" "$RUNNER" \
    --analysis-tier formal \
    --smoke-manifest "$smoke_out/mech02_manifest.json" \
    --python-exe "$TRAIN_PY" \
    --family "$family" \
    --method "$method" \
    --checkpoint "$checkpoint" \
    --checkpoint-sha256 "$checkpoint_sha" \
    --hash-checkpoint \
    --source-script "$source" \
    --triton-kernels "$triton" \
    --mech01-smoke-dir "$mech01" \
    --data-pattern "$data_pattern" \
    --host-id llama-host-h100 \
    --execution-domain "llama-host-$family" \
    --output-dir "$formal_out" \
    --run-prefix "${slug}_formal"

  assert_passed "$formal_out/mech02_manifest.json"
  echo "MECH-02 family PASS: $family"
}

export CUDA_VISIBLE_DEVICES="$MECH_GPU"
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"

run_family \
  gpt_bridge_endpoint gpt_bridge none \
  "$GPT_CHECKPOINT" "$GPT_SHA256" "$GPT_SOURCE" "$GPT_TRITON" \
  "$BRIDGE_DATA" "$GPT_MECH01"

run_family \
  llama124_endpoint llama124 down_none \
  "$L124_CHECKPOINT" "$L124_SHA256" "$L124_SOURCE" "$LLAMA_TRITON" \
  "$LLAMA_DATA" "$L124_MECH01"

PACKAGE_DIR="${SNM_RESULTS_ROOT}/30_mech02_k_geometry/handoff_packages"
PACKAGE="$PACKAGE_DIR/mech02_llama_host_endpoint_${STAMP}.tgz"
mkdir -p "$PACKAGE_DIR"
tar -C "$(dirname "$ROOT")" -czf "$PACKAGE" "$(basename "$ROOT")"
echo "MECH-02 LLaMA-host endpoint PASS"
echo "MECH-02 LLaMA-host handoff: $PACKAGE"
sha256sum "$PACKAGE"
ls -lh "$PACKAGE"
