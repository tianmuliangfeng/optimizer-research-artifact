#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCHER="${ROOT}/commands/29_r1_depth_kmode/20260806_ex29_r1_depth_kmode.sh"

export SNM_ARTIFACT_ROOT="${SNM_ARTIFACT_ROOT:-${ROOT}}"
export SNM_REPO="${SNM_REPO:-${ROOT}}"
export SNM_RESULTS_ROOT="${SNM_RESULTS_ROOT:-${ROOT}/runs}"
export SNM_OFFICIAL_REPO="${SNM_OFFICIAL_REPO:-${ROOT}/third_party/Newton-Muon-official-r0}"
export EX29_RESULTS_DIR="${EX29_RESULTS_DIR:-${SNM_RESULTS_ROOT}/29_r1_depth_kmode/results}"

bash "${LAUNCHER}" check
bash "${LAUNCHER}" preflight
bash "${LAUNCHER}" formal

