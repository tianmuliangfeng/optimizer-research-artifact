#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCHER="${ROOT}/commands/49_r1_malt_strong_baseline/20260807_ex49_r1_malt_strong_baseline.sh"
STAMP="$(date -u +%Y%m%dT%H%M%S+0000)"

export SNM_ARTIFACT_ROOT="${SNM_ARTIFACT_ROOT:-${ROOT}}"
export SNM_REPO="${SNM_REPO:-${ROOT}}"
export SNM_RESULTS_ROOT="${SNM_RESULTS_ROOT:-${ROOT}/runs}"
export SNM_OFFICIAL_REPO="${SNM_OFFICIAL_REPO:-${ROOT}/third_party/Newton-Muon-official-r0}"
export EX49_RUN_DIR="${EX49_RUN_DIR:-${SNM_RESULTS_ROOT}/49_r1_malt_strong_baseline/${STAMP}}"

# The suite's historical `all` mode resumes accepted units but does not invoke
# the runtime/data preflight. A fresh public reproduction therefore executes
# every gate explicitly in the frozen order.
bash "${LAUNCHER}" preflight
bash "${LAUNCHER}" pilot
bash "${LAUNCHER}" formal
bash "${LAUNCHER}" verify

