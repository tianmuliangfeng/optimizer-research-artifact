#!/usr/bin/env python3
"""Certify the pinned, clean official repository used by experiment 39."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-07-29.1"
SCRIPT_DIR = Path(__file__).resolve().parent
R0_CONTROLLER = (
    SCRIPT_DIR.parent
    / "14_official_newton_muon_r0"
    / "run_official_newton_muon_r0.py"
)


def load_controller() -> Any:
    spec = importlib.util.spec_from_file_location(
        "experiment39_r0_provenance", R0_CONTROLLER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import provenance controller: {R0_CONTROLLER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    controller = load_controller()
    provenance = controller.validate_official_repo(
        args.official_repo.expanduser().resolve()
    )
    payload = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "provenance": provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Official-repository certificate: {args.output}")


if __name__ == "__main__":
    main()
