#!/usr/bin/env python3
"""Outcome-blind controller front end for experiment 47 / GEO-01.

This lightweight entry point exposes contract checking, a local sealed dry-run
and a CPU toy pilot. Remote H100 orchestration lives in ``remote_controller.py``
so a local dry-run cannot accidentally launch checkpoint work.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
sys.path.insert(0, str(HERE))

import protocol as P


SCRIPT_VERSION = "2026-08-04.3"
SOURCE_FILES = (
    "geometry_core.py",
    "geo01_worker.py",
    "protocol.py",
    "geo01_contract.json",
    "run_geo01.py",
    "remote_controller.py",
    "analyze_geo01.py",
    "test_geo01.py",
    "README.md",
)


def source_inventory() -> dict[str, Any]:
    files = {}
    for name in SOURCE_FILES:
        path = HERE / name
        if not path.is_file():
            raise FileNotFoundError(f"required GEO-01 source is missing: {path}")
        files[name] = {
            "bytes": path.stat().st_size,
            "sha256": P.sha256_file(path),
        }
    return {
        "schema_version": "geo01_live_source_inventory_v1",
        "controller_version": SCRIPT_VERSION,
        "files": files,
        "passed": True,
    }


def run_check(contract_path: Path) -> dict[str, Any]:
    validation = P.validation_payload(contract_path)
    validation["source_inventory"] = source_inventory()
    validation["controller_capabilities"] = {
        "check": True,
        "dry_run": True,
        "toy_pilot": True,
        "remote_h100_pilot": True,
        "discovery": False,
        "confirmation": False,
        "llama_10b": False,
    }
    validation["passed"] = (
        validation["passed"]
        and validation["source_inventory"]["passed"]
    )
    return validation


def snapshot_sources(target: Path) -> dict[str, Any]:
    target.mkdir(parents=True, exist_ok=False)
    files = {}
    for name in SOURCE_FILES:
        source = HERE / name
        destination = target / name
        shutil.copy2(source, destination)
        files[name] = P.sha256_file(destination)
    payload = {
        "schema_version": "geo01_source_snapshot_v1",
        "controller_version": SCRIPT_VERSION,
        "files": files,
        "passed": all(
            P.sha256_file(target / name) == expected
            for name, expected in files.items()
        ),
    }
    P.atomic_json(target / "source_snapshot_manifest.json", payload)
    return payload


def run_dry_run(contract_path: Path, run_dir: Path) -> dict[str, Any]:
    check = run_check(contract_path)
    if not check["passed"]:
        raise RuntimeError(f"GEO-01 contract check failed: {check['checks']}")
    run_dir.mkdir(parents=True, exist_ok=False)
    identity = {
        "schema_version": "geo01_run_identity_v1",
        "experiment": "GEO-01",
        "experiment_number": 47,
        "phase": "pilot_dry_run",
        "controller_version": SCRIPT_VERSION,
        "contract_sha256": P.sha256_file(contract_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "remote_execution_authorized": False,
    }
    P.atomic_json(run_dir / "run_identity.json", identity)
    P.atomic_json(run_dir / "contract_validation.json", check)
    snapshot = snapshot_sources(run_dir / "source_snapshot")
    plan = {
        "schema_version": "geo01_pilot_plan_v1",
        "phase": "pilot",
        "origin": "early_muon",
        "data_replica": 7,
        "event_id": "production_refresh_32",
        "scopes": ["layer_0", "layer_8", "layer_17", "joint_0_8_17"],
        "scientific_outcome_may_be_opened": False,
        "remote_worker_available": True,
        "remote_worker_entrypoint": "remote_controller.py",
        "passed": True,
    }
    P.atomic_json(run_dir / "pilot_plan.json", plan)
    status = {
        "status": "dry_run_passed_remote_launch_blocked",
        "controller_version": SCRIPT_VERSION,
        "contract_sha256": identity["contract_sha256"],
        "source_snapshot_passed": snapshot["passed"],
        "passed": check["passed"] and snapshot["passed"],
    }
    P.atomic_json(run_dir / "status.json", status)
    return status


def run_toy_pilot(contract_path: Path, run_dir: Path) -> dict[str, Any]:
    check = run_check(contract_path)
    if not check["passed"]:
        raise RuntimeError("contract check failed before toy pilot")
    import torch
    from torch import nn

    import geometry_core as G

    torch.manual_seed(20260804)

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(3, 4, dtype=torch.float64) / 5.0)

        def forward(self, x: torch.Tensor, y: torch.Tensor, **_: Any) -> tuple[Tensor, Tensor]:
            prediction = x @ self.weight.t()
            loss = (prediction - y).square().mean()
            return prediction, loss

    from torch import Tensor

    model = ToyModel()
    batches = [
        (
            torch.randn(5, 4, dtype=torch.float64),
            torch.randn(5, 3, dtype=torch.float64),
        ),
        (
            torch.randn(4, 4, dtype=torch.float64),
            torch.randn(4, 3, dtype=torch.float64),
        ),
    ]
    direction = {"weight": torch.randn_like(model.weight) * 1.0e-3}
    result = G.measure_directional_geometry(
        model=model,
        batches=batches,
        named_direction=direction,
        fd_target_relative_parameter_norm=1.0e-4,
        fd_scale_min=1.0,
        fd_scale_max=64.0,
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    P.atomic_json(run_dir / "toy_geometry.json", result)
    status = {
        "schema_version": "geo01_toy_pilot_status_v1",
        "status": "passed" if (
            result["all_values_finite"]
            and result["parameters_unchanged"]
            and result["fd_first_relative_error"] <= 1.0e-7
            and result["fd_curvature_relative_error"] <= 1.0e-6
            and abs(result["taylor_residual"]) <= 1.0e-10
        ) else "failed",
        "contract_sha256": P.sha256_file(contract_path),
        "geometry": result,
    }
    status["passed"] = status["status"] == "passed"
    P.atomic_json(run_dir / "status.json", status)
    if not status["passed"]:
        raise RuntimeError("toy directional-geometry pilot failed")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("check", "dry-run", "toy-pilot")
    )
    parser.add_argument(
        "--contract", type=Path, default=HERE / "geo01_contract.json"
    )
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = args.contract.resolve()
    if args.mode == "check":
        result = run_check(contract)
    else:
        if args.run_dir is None:
            raise ValueError(f"{args.mode} requires --run-dir")
        if args.mode == "dry-run":
            result = run_dry_run(contract, args.run_dir.resolve())
        else:
            result = run_toy_pilot(contract, args.run_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result.get("passed") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
