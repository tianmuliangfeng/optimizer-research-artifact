#!/usr/bin/env bash
set -euo pipefail

_snm_search_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ "${_snm_search_dir}" != "/" && ! -f "${_snm_search_dir}/pyproject.toml" ]]; do
  _snm_search_dir="$(dirname -- "${_snm_search_dir}")"
done
SNM_ARTIFACT_ROOT="${SNM_ARTIFACT_ROOT:-${_snm_search_dir}}"
SNM_REPO="${SNM_REPO:-${SNM_ARTIFACT_ROOT}}"
_snm_default_python="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
SNM_CONTROLLER_PYTHON="${SNM_CONTROLLER_PYTHON:-${_snm_default_python:-python3}}"

# Analysis-only repair for the already completed EX55 seed-2027 formal run.
# This command never invokes pilot/formal training; it reruns only verification
# and final analysis against the existing ten accepted formal artifacts.

EX55_REPO="${EX55_REPO:-${SNM_REPO}}"
EX55_CONTROLLER_PYTHON="${EX55_CONTROLLER_PYTHON:-${SNM_CONTROLLER_PYTHON}}"
EX55_RUN_DIR="${1:-${EX55_RUN_DIR:-}}"
if [[ -z "${EX55_RUN_DIR}" ]]; then
  echo "usage: $0 /path/to/existing/ex55-run" >&2
  exit 2
fi
EX55_WRAPPER="${EX55_REPO}/commands/55_r1_fresh_seed_baseline_fairness/20260817_ex55_r1_fresh_seed_baseline_fairness.sh"
EX55_SCRIPT_ROOT="${EX55_REPO}/scripts/55_r1_fresh_seed_baseline_fairness"

for path in \
  "${EX55_WRAPPER}" \
  "${EX55_SCRIPT_ROOT}/analyze_fresh_seed_panel.py" \
  "${EX55_SCRIPT_ROOT}/run_fresh_seed_suite.py" \
  "${EX55_RUN_DIR}/formal_units_manifest.json" \
  "${EX55_RUN_DIR}/source_snapshot/source_snapshot_manifest.json"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required EX55 file is absent: ${path}" >&2
    exit 2
  fi
done
if [[ ! -x "${EX55_CONTROLLER_PYTHON}" ]]; then
  echo "Controller Python is not executable: ${EX55_CONTROLLER_PYTHON}" >&2
  exit 2
fi
if ! grep -q 'FORMAL_METRICS_TAIL5_LINEAGE = True' \
  "${EX55_SCRIPT_ROOT}/analyze_fresh_seed_panel.py"; then
  echo "The live analyzer does not contain the formal-metrics tail-5 repair." >&2
  exit 2
fi
if ! grep -q 'FORMAL_METRICS_TAIL5_LINEAGE = True' \
  "${EX55_SCRIPT_ROOT}/run_fresh_seed_suite.py"; then
  echo "The live controller does not contain the formal-metrics lineage verifier." >&2
  exit 2
fi

"${EX55_CONTROLLER_PYTHON}" -B - "${EX55_RUN_DIR}/formal_units_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
units = payload.get("units")
methods = {str(record.get("method", "")) for record in units or [] if isinstance(record, dict)}
expected = {
    "block4", "diag", "none", "muon", "adamw", "normuon", "moonlight",
    "mousse", "malt", "malter_eq17",
}
if (
    payload.get("passed") is not True
    or payload.get("formal_seed") != 2027
    or payload.get("formal_units") != 10
    or not isinstance(units, list)
    or len(units) != 10
    or methods != expected
):
    raise SystemExit(f"formal_units_manifest.json is incomplete: methods={sorted(methods)}")
print("Formal evidence gate: 10/10 seed-2027 units are present.")
PY

cd "${EX55_REPO}"
"${EX55_CONTROLLER_PYTHON}" -B \
  "${EX55_SCRIPT_ROOT}/test_analyze_fresh_seed_panel.py"
"${EX55_CONTROLLER_PYTHON}" -B \
  "${EX55_SCRIPT_ROOT}/test_fresh_seed_suite.py"

echo "Starting EX55 analysis-only verify; no training stage will be launched."
EX55_RUN_DIR="${EX55_RUN_DIR}" \
EX55_REPO="${EX55_REPO}" \
EX55_CONTROLLER_PYTHON="${EX55_CONTROLLER_PYTHON}" \
EX55_GPUS="1" \
EX55_WANDB_MODE="disabled" \
  bash "${EX55_WRAPPER}" verify

echo "EX55 repaired analysis manifest: ${EX55_RUN_DIR}/analysis/analysis_manifest.json"
echo "EX55 metrics lineage: ${EX55_RUN_DIR}/analysis/formal_metrics_tail5_lineage.json"
echo "EX55 handoff manifest: ${EX55_RUN_DIR}/handoff_manifest.json"
