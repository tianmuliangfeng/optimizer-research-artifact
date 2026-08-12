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

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SNM_REPO}}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${SNM_RESULTS_ROOT}}"
CTRL_PY="${CTRL_PY:-${SNM_CONTROLLER_PYTHON}}"
RUNNER="$PROJECT_ROOT/scripts/26_mech00_inventory/run_mech00_inventory.py"
FOLLOWUP_ROOT="$EXPERIMENT_ROOT/20_llama_swiglu_1b/multiseed_followup"
STAMP="$(date -u +%Y%m%dT%H%M%S+0000)"
OUT_DIR="${MECH00_OUTPUT_DIR:-$EXPERIMENT_ROOT/26_mech00_inventory/llama_host_followup_${STAMP}_full}"
VALIDATE_ONLY="${MECH00_VALIDATE_ONLY:-0}"

if [[ ! -f "$CTRL_PY" || ! -f "$RUNNER" || ! -d "$FOLLOWUP_ROOT" ]]; then
  echo "Missing controller, runner, or multiseed_followup directory." >&2
  exit 2
fi

"$CTRL_PY" - "$RUNNER" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(r'^SCRIPT_VERSION\s*=\s*"([^"]+)"', text, re.MULTILINE)
assert match and match.group(1) == "2026-07-24.2", (
    "Sync MECH-00 v2026-07-24.2 before hashing; "
    f"observed={match.group(1) if match else 'missing'}"
)
PY

if [[ "$VALIDATE_ONLY" == "1" ]]; then
  echo "Validating existing supplemental MECH-00 output without rehashing."
  echo "Output: $OUT_DIR"
else
  echo "Starting supplemental LLaMA-1B multiseed full hash."
  echo "Output: $OUT_DIR"

  set +e
  "$CTRL_PY" "$RUNNER" \
    --host-id llama-host-h100 \
    --execution-domain llama-host \
    --input "llama1b_followup=$FOLLOWUP_ROOT" \
    --family-hint llama1b_followup=llama_1b \
    --repo "llama_source=${SNM_OFFICIAL_REPO}" \
    --methods down_none muon \
    --target-steps 6200 \
    --hash-mode full \
    --output-dir "$OUT_DIR" \
    --strict &
  AUDIT_PID=$!

  while kill -0 "$AUDIT_PID" 2>/dev/null; do
    echo "MECH-00 followup heartbeat: $(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$AUDIT_PID"
    sleep 60
  done
  wait "$AUDIT_PID"
  STATUS=$?
  set -e

  if [[ "$STATUS" -ne 0 ]]; then
    echo "Supplemental MECH-00 runner failed with exit code $STATUS" >&2
    exit "$STATUS"
  fi
fi

"$CTRL_PY" - "$OUT_DIR" <<'PY'
import csv
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
manifest = json.loads((out / "mech00_manifest.json").read_text(encoding="utf-8"))
with (out / "checkpoint_hashes.csv").open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

required_formal = {
    ("2024", "down_none", "6200"),
    ("2024", "muon", "6200"),
    ("2025", "down_none", "6200"),
    ("2025", "muon", "6200"),
}
allowed_medium = {
    ("2024", "down_none", "1000"),
    ("2024", "muon", "1000"),
    ("2025", "down_none", "1000"),
    ("2025", "muon", "1000"),
}
expected_all = required_formal | allowed_medium
observed = {(row["seed"], row["method"], row["completed_steps"]) for row in rows}
assert manifest["script_version"] == "2026-07-24.2", manifest
assert manifest["hash_mode"] == "full", manifest
assert manifest["counts"]["failed_checks"] == 0, manifest["counts"]
assert observed == expected_all, {
    "expected": sorted(expected_all),
    "observed": sorted(observed),
}
assert len(rows) == len(observed) == 8, {
    "rows": len(rows),
    "unique_seed_method_step": len(observed),
}
assert all(
    row["hash_status"] == "verified_stable" and len(row["sha256"]) == 64
    for row in rows
), rows
print(
    "MECH-00 supplemental gate passed: "
    "checkpoints=8 formal_step6200=4 medium_step1000=4"
)
print(f"MECH00_FOLLOWUP_MANIFEST={out / 'mech00_manifest.json'}")
print(f"MECH00_FOLLOWUP_HASHES={out / 'checkpoint_hashes.csv'}")
PY
