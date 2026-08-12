"""Dedicated entry point for the GPT R1 diag/none host bridge.

The implementation intentionally reuses the audited R1 source builder and
evidence gates. This wrapper only selects the separate host-bridge family; the
underlying runner enforces the diag/none method set and timing-ineligible
evidence policy.
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
R1_DIR = SCRIPT_DIR.parent / "15_official_newton_muon_r1"
sys.path.insert(0, str(R1_DIR))

import run_official_newton_muon_r1 as r1


REQUIRED_R1_BRIDGE_API = (
    "HOST_BRIDGE_FAMILY",
    "HOST_BRIDGE_FORMAL_PROTOCOL",
    "HOST_BRIDGE_SMOKE_PROTOCOL",
    "evidence_eligibility",
    "visible_device_record",
)


def require_bridge_capable_r1() -> None:
    missing = [name for name in REQUIRED_R1_BRIDGE_API if not hasattr(r1, name)]
    if missing:
        raise RuntimeError(
            "The imported scripts/15_official_newton_muon_r1/"
            "run_official_newton_muon_r1.py is stale and cannot produce auditable "
            "host-bridge evidence. Sync the updated base runner together with this "
            f"wrapper, then retry. Missing bridge API: {', '.join(missing)}"
        )


def bridge_argv(argv: list[str]) -> list[str]:
    if "--host-bridge" in argv[1:]:
        return argv
    return [argv[0], "--host-bridge", *argv[1:]]


def main() -> None:
    require_bridge_capable_r1()
    sys.argv = bridge_argv(sys.argv)
    r1.main()


if __name__ == "__main__":
    main()
