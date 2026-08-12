#!/usr/bin/env python3
"""CPU-only contract and source check for experiment 47 / GEO-01B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import protocol as P


SCRIPT_VERSION = "2026-08-04.1"
LOCAL_FILES = (
    "README.md",
    "geo01b_contract.json",
    "protocol.py",
    "geo01b_worker.py",
    "analyze_geo01b.py",
    "remote_controller.py",
    "run_geo01b.py",
    "test_geo01b.py",
)


def source_inventory() -> dict[str, Any]:
    rows = {}
    for name in LOCAL_FILES:
        path = HERE / name
        if not path.is_file():
            raise FileNotFoundError(f"required GEO-01B source is missing: {path}")
        rows[name] = {"bytes": path.stat().st_size, "sha256": P.sha256_file(path)}
    launcher = REPO / "commands/47b_geo01b_update_geometry_discovery/20260804_ex47b_geo01b_discovery.sh"
    if not launcher.is_file():
        raise FileNotFoundError(f"GEO-01B launcher is missing: {launcher}")
    return {
        "schema_version": "geo01b_live_source_inventory_v1",
        "files": rows,
        "launcher_sha256": P.sha256_file(launcher),
        "passed": True,
    }


def check(contract_path: Path) -> dict[str, Any]:
    result = P.validation_payload(contract_path)
    result["controller_version"] = SCRIPT_VERSION
    result["source_inventory"] = source_inventory()
    result["job_matrix"] = P.job_matrix(P.read_json(contract_path))
    result["capabilities"] = {
        "local_check": True,
        "sealed_remote_dry_run": True,
        "remote_smoke": True,
        "remote_discovery": True,
        "same_contract_resume": True,
        "read_only_verify": True,
        "confirmation": False,
        "llama_10b": False,
    }
    result["passed"] = (
        result["passed"]
        and result["source_inventory"]["passed"]
        and len(result["job_matrix"]) == 12
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check",))
    parser.add_argument(
        "--contract", type=Path, default=HERE / "geo01b_contract.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = check(args.contract.resolve())
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
