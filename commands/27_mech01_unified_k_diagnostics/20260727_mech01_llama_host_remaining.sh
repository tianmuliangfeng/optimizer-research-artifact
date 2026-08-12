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

# Complete the remaining MECH-01 implementation gates on the LLaMA H100 host.
# The three families run sequentially on one GPU:
#   1. GPT bridge none seed2026 @ step6200
#   2. LLaMA-124M down_none seed2026 @ step6200
#   3. LLaMA-1B down_none seed2026 @ step6200
#
# Every family starts from preflight, re-checks the MECH-00 full checkpoint
# SHA-256, and only then enters numerical smoke. No stage trains, calls
# optimizer.step(), writes a checkpoint, or uploads to W&B.

PROJECT_ROOT="${PROJECT_ROOT:-${SNM_REPO}}"
CTRL_PY="${CTRL_PY:-${SNM_CONTROLLER_PYTHON}}"
TRAIN_PY="${TRAIN_PY:-${SNM_TRAINING_PYTHON}}"
MECH_GPU="${MECH_GPU:-0}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-60}"

RUNNER="$PROJECT_ROOT/scripts/27_mech01_unified_k_diagnostics/run_mech01.py"
LLAMA_REPO="${SNM_OFFICIAL_REPO}"
BRIDGE_REPO="${SNM_OFFICIAL_REPO}"
LLAMA_TRITON="$LLAMA_REPO/triton_kernels.py"
LLAMA_DATA_PATTERN="$LLAMA_REPO/data/fineweb10B/fineweb_val_*.bin"
BRIDGE_DATA_PATTERN="$BRIDGE_REPO/data/fineweb10B/fineweb_val_*.bin"

GPT_WORKSPACE="${SNM_RESULTS_ROOT}/21_gpt_r1_host_bridge/results/20260721T072708+0000_formal_seed2026/mainconf_gpt_r1_host_bridge_none_seed2026_20260721T072708+0000/workspace"
GPT_CHECKPOINT="$GPT_WORKSPACE/logs/ba9acc67-9a51-4fea-b1bc-695e11bb9c80/state_step006200.pt"
GPT_SOURCE="$GPT_WORKSPACE/train_r1_none.py"
GPT_TRITON="$GPT_WORKSPACE/triton_kernels.py"

L124_RUN="${SNM_RESULTS_ROOT}/17_llama_swiglu_validation/results/20260720T062421+0000_formal_seed2026/02_down_none"
L124_CHECKPOINT="$L124_RUN/checkpoint_latest.pt"
L124_SOURCE="$L124_RUN/train_llama_swiglu.py"

L1B_RUN="${SNM_RESULTS_ROOT}/20_llama_swiglu_1b/formal/gpu0_none_diag/20260723T025322+0000_formal_seed2026/01_down_none"
L1B_CHECKPOINT="$L1B_RUN/checkpoint_latest.pt"
L1B_SOURCE="$L1B_RUN/train_llama_swiglu_base.py"
L1B_PROFILE="$L1B_RUN/train_llama_swiglu.py"

EXPECTED_GPT_CHECKPOINT_BYTES=1312617750
EXPECTED_GPT_CHECKPOINT_SHA256="bb9a7d2dcc191cd72caf01e24b838f99c4c319ca3dfdfff036684b939bacef6c"
EXPECTED_GPT_SOURCE_SHA256="d4368715bb64d6ad89509876c4d7f28773fa4dd567accca09405ff34e74220f5"
EXPECTED_GPT_TRITON_SHA256="b51ac50c699b05306619d92cb9ec6edadd266d8118c53f5b9726db76480ea16d"

EXPECTED_L124_CHECKPOINT_BYTES=1399146415
EXPECTED_L124_CHECKPOINT_SHA256="f13f8b7bd43de31f1f2190702b4954d707bea5489d728ce673145979881682fd"
EXPECTED_L124_SOURCE_SHA256="b72eb0d2a1dfa91b61cd49b4784b3e0739ecebc2fd3228b8f719cec125706f2a"

EXPECTED_L1B_CHECKPOINT_BYTES=11240324367
EXPECTED_L1B_CHECKPOINT_SHA256="069bc086a79bd71c3723cabeb9f346730b931610c703b3725a7031ad4d1fa8f0"
EXPECTED_L1B_SOURCE_SHA256="$EXPECTED_L124_SOURCE_SHA256"
EXPECTED_L1B_PROFILE_SHA256="043c758f3d5eb5d1abc9e1f9029a8d085a238cf169ef69ba86580014699dc401"

EXPECTED_LLAMA_TRITON_SHA256="f092ae994f5a5c1ebacf3938e2bb8d610dc537b928e2a5039438afe1e46a271f"

STAMP="$(date -u +%Y%m%dT%H%M%S+0000)"
RESULTS_ROOT="${SNM_RESULTS_ROOT}/27_mech01_unified_k_diagnostics/llama_host_remaining/$STAMP"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "MECH-01 missing required file: $1" >&2
    exit 2
  fi
}

require_glob() {
  if ! compgen -G "$1" >/dev/null; then
    echo "MECH-01 data pattern has no files: $1" >&2
    exit 2
  fi
}

require_size() {
  local path="$1"
  local expected="$2"
  local observed
  observed="$(stat -c '%s' "$path")"
  if [[ "$observed" != "$expected" ]]; then
    echo "MECH-01 file size mismatch: path=$path observed=$observed expected=$expected" >&2
    exit 2
  fi
}

require_sha256() {
  local path="$1"
  local expected="$2"
  local observed
  observed="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$observed" != "$expected" ]]; then
    echo "MECH-01 SHA-256 mismatch: path=$path observed=$observed expected=$expected" >&2
    exit 2
  fi
}

run_with_heartbeat() {
  "$@" &
  local child_pid=$!
  while kill -0 "$child_pid" 2>/dev/null; do
    sleep "$HEARTBEAT_SECONDS"
    if kill -0 "$child_pid" 2>/dev/null; then
      echo "MECH-01 heartbeat $(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$child_pid gpu=$MECH_GPU"
      nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null || true
    fi
  done
  wait "$child_pid"
}

assert_manifest_passed() {
  "$CTRL_PY" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("passed") is True, d' \
    "$1"
}

checkpoint_sha_from_preflight() {
  "$CTRL_PY" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert d["checkpoint_hash_checked"] and d["checkpoint_hash_pass"]; print(d["checkpoint_sha256_observed"])' \
    "$1"
}

run_family_gate() {
  local slug="$1"
  local family="$2"
  local method="$3"
  local checkpoint="$4"
  local expected_checkpoint_sha="$5"
  local source="$6"
  local triton="$7"
  local data_pattern="$8"
  local profile="${9:-}"

  local preflight_out="$RESULTS_ROOT/$slug/preflight"
  local smoke_out="$RESULTS_ROOT/$slug/numerical_smoke"
  local layers=()
  local candidates=()
  local bundle_layer
  local profile_args=()

  case "$family" in
    gpt_bridge)
      layers=(0 6 11)
      candidates=(none diag block4 dense_full)
      bundle_layer=6
      ;;
    llama124)
      layers=(0 6 11)
      candidates=(none diag dense_full)
      bundle_layer=6
      ;;
    llama1b)
      layers=(0 9 17)
      candidates=(none diag dense_full)
      bundle_layer=9
      ;;
    *)
      echo "MECH-01 unsupported family in remaining gate: $family" >&2
      exit 2
      ;;
  esac

  if [[ -n "$profile" ]]; then
    profile_args=(--profile-script "$profile")
  fi

  echo "MECH-01 $family preflight output: $preflight_out"
  run_with_heartbeat \
    "$CTRL_PY" "$RUNNER" \
    --preflight \
    --python-exe "$TRAIN_PY" \
    --family "$family" \
    --checkpoint "$checkpoint" \
    --source-script "$source" \
    --triton-kernels "$triton" \
    "${profile_args[@]}" \
    --method "$method" \
    --host-id llama-host-h100 \
    --execution-domain "llama-host-$family" \
    --hash-checkpoint \
    --output-dir "$preflight_out" \
    --run-prefix "${slug}_preflight"

  assert_manifest_passed "$preflight_out/mech01_manifest.json"

  local checkpoint_sha
  checkpoint_sha="$(checkpoint_sha_from_preflight "$preflight_out/checkpoint_schema.json")"
  if [[ "$checkpoint_sha" != "$expected_checkpoint_sha" ]]; then
    echo "MECH-01 checkpoint differs from MECH-00: family=$family observed=$checkpoint_sha expected=$expected_checkpoint_sha" >&2
    exit 2
  fi

  echo "MECH-01 $family numerical smoke output: $smoke_out"
  run_with_heartbeat \
    "$CTRL_PY" "$RUNNER" \
    --numerical-smoke \
    --python-exe "$TRAIN_PY" \
    --family "$family" \
    --checkpoint "$checkpoint" \
    --source-script "$source" \
    --triton-kernels "$triton" \
    "${profile_args[@]}" \
    --method "$method" \
    --data-pattern "$data_pattern" \
    --layers "${layers[@]}" \
    --device-batch-size 1 \
    --sequence-length 128 \
    --probe-offsets 0 4096 8192 12288 \
    --max-activation-rows 2048 \
    --ridge-mult 0.2 \
    --ridge-eps 1e-8 \
    --momentum 0.95 \
    --ns-steps 5 \
    --candidates "${candidates[@]}" \
    --spectrum-dtype float64 \
    --export-bundle-layer "$bundle_layer" \
    --checkpoint-sha256 "$checkpoint_sha" \
    --hash-checkpoint \
    --host-id llama-host-h100 \
    --execution-domain "llama-host-$family" \
    --output-dir "$smoke_out" \
    --run-prefix "${slug}_smoke"

  assert_manifest_passed "$smoke_out/mech01_manifest.json"
  echo "MECH-01 family PASS: $family"
  echo "MECH-01 family manifest: $smoke_out/mech01_manifest.json"
}

require_file "$CTRL_PY"
require_file "$TRAIN_PY"
require_file "$RUNNER"
require_file "$GPT_CHECKPOINT"
require_file "$GPT_SOURCE"
require_file "$GPT_TRITON"
require_file "$L124_CHECKPOINT"
require_file "$L124_SOURCE"
require_file "$L1B_CHECKPOINT"
require_file "$L1B_SOURCE"
require_file "$L1B_PROFILE"
require_file "$LLAMA_TRITON"
require_glob "$BRIDGE_DATA_PATTERN"
require_glob "$LLAMA_DATA_PATTERN"

require_size "$GPT_CHECKPOINT" "$EXPECTED_GPT_CHECKPOINT_BYTES"
require_size "$L124_CHECKPOINT" "$EXPECTED_L124_CHECKPOINT_BYTES"
require_size "$L1B_CHECKPOINT" "$EXPECTED_L1B_CHECKPOINT_BYTES"

require_sha256 "$GPT_SOURCE" "$EXPECTED_GPT_SOURCE_SHA256"
require_sha256 "$GPT_TRITON" "$EXPECTED_GPT_TRITON_SHA256"
require_sha256 "$L124_SOURCE" "$EXPECTED_L124_SOURCE_SHA256"
require_sha256 "$L1B_SOURCE" "$EXPECTED_L1B_SOURCE_SHA256"
require_sha256 "$L1B_PROFILE" "$EXPECTED_L1B_PROFILE_SHA256"
require_sha256 "$LLAMA_TRITON" "$EXPECTED_LLAMA_TRITON_SHA256"

grep 'SCRIPT_VERSION =' \
  "$PROJECT_ROOT/scripts/27_mech01_unified_k_diagnostics/mech01_worker.py"

export CUDA_VISIBLE_DEVICES="$MECH_GPU"
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"

run_family_gate \
  "gpt_bridge_seed2026_none6200" \
  "gpt_bridge" \
  "none" \
  "$GPT_CHECKPOINT" \
  "$EXPECTED_GPT_CHECKPOINT_SHA256" \
  "$GPT_SOURCE" \
  "$GPT_TRITON" \
  "$BRIDGE_DATA_PATTERN"

run_family_gate \
  "llama124_seed2026_down_none6200" \
  "llama124" \
  "down_none" \
  "$L124_CHECKPOINT" \
  "$EXPECTED_L124_CHECKPOINT_SHA256" \
  "$L124_SOURCE" \
  "$LLAMA_TRITON" \
  "$LLAMA_DATA_PATTERN"

run_family_gate \
  "llama1b_seed2026_down_none6200" \
  "llama1b" \
  "down_none" \
  "$L1B_CHECKPOINT" \
  "$EXPECTED_L1B_CHECKPOINT_SHA256" \
  "$L1B_SOURCE" \
  "$LLAMA_TRITON" \
  "$LLAMA_DATA_PATTERN" \
  "$L1B_PROFILE"

echo "MECH-01 remaining families PASS"
echo "MECH-01 remaining artifacts: $RESULTS_ROOT"

PACKAGE_DIR="${SNM_RESULTS_ROOT}/27_mech01_unified_k_diagnostics/handoff_packages"
PACKAGE="$PACKAGE_DIR/mech01_llama_host_remaining_${STAMP}.tgz"
mkdir -p "$PACKAGE_DIR"
tar -C "$(dirname "$RESULTS_ROOT")" \
  -czf "$PACKAGE" \
  "$(basename "$RESULTS_ROOT")"
echo "MECH-01 remaining handoff package: $PACKAGE"
sha256sum "$PACKAGE"
ls -lh "$PACKAGE"
