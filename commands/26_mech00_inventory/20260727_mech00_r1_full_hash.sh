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
STAMP="$(date -u +%Y%m%dT%H%M%S+0000)"
OUT_DIR="$EXPERIMENT_ROOT/26_mech00_inventory/r1_native_${STAMP}_full"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 2
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
    exit 2
  fi
}

require_file "$CTRL_PY"
require_file "$RUNNER"
require_dir "$EXPERIMENT_ROOT/15_official_newton_muon_r1/results"
require_dir "${SNM_OFFICIAL_REPO}"

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

echo "Starting read-only MECH-00 full hash on the R1-native host."
echo "Output: $OUT_DIR"

set +e
"$CTRL_PY" "$RUNNER" \
  --host-id r1-native-h100 \
  --execution-domain r1-native \
  --input "r1=$EXPERIMENT_ROOT/15_official_newton_muon_r1/results" \
  --family-hint r1=r1_native \
  --repo "r1_source=${SNM_OFFICIAL_REPO}" \
  --methods none muon \
  --target-steps 0 500 1000 3000 6200 \
  --hash-mode full \
  --output-dir "$OUT_DIR" \
  --strict &
AUDIT_PID=$!

while kill -0 "$AUDIT_PID" 2>/dev/null; do
  echo "MECH-00 heartbeat: $(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$AUDIT_PID"
  sleep 60
done
wait "$AUDIT_PID"
STATUS=$?
set -e

if [[ "$STATUS" -ne 0 ]]; then
  echo "MECH-00 runner failed with exit code $STATUS" >&2
  exit "$STATUS"
fi

"$CTRL_PY" - "$OUT_DIR" <<'PY'
import csv
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
manifest = json.loads((out / "mech00_manifest.json").read_text(encoding="utf-8"))
rows = list(csv.DictReader((out / "checkpoint_hashes.csv").open(encoding="utf-8", newline="")))
assert manifest["hash_mode"] == "full", manifest
assert manifest["counts"]["failed_checks"] == 0, manifest["counts"]
assert rows, "no checkpoint hash rows were produced"
assert len(rows) == int(manifest["counts"]["checkpoint_files"])
bad = [
    row for row in rows
    if row["hash_status"] != "verified_stable" or len(row["sha256"]) != 64
]
assert not bad, f"unverified checkpoint hashes: {bad}"
print(f"MECH-00 full-hash gate passed: checkpoints={len(rows)}")
print(f"MECH00_FULL_MANIFEST={out / 'mech00_manifest.json'}")
print(f"MECH00_CHECKPOINT_HASHES={out / 'checkpoint_hashes.csv'}")
PY
