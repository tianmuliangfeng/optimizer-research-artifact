"""Run the predeclared odd-batch confirmation of the 1B OOM boundary.

This controller consumes a completed even-grid fine-capacity manifest.  For
each requested method whose endpoints differ by exactly two, it runs only the
single integer batch between the successful and OOM endpoints, then combines
that result with the parent endpoints to produce an exact integer boundary.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_llama_swiglu_1b_capacity_fine as fine


PROTOCOL = "microbatch_capacity_exact_odd_confirmation_v1"
METHOD_ORDER = ("newton_full", "down_none", "down_diag", "muon")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predeclared odd-batch confirmation for the 1B fine OOM sweep"
    )
    parser.add_argument("--fine-manifest", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--python-exe", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=sorted(fine.METHODS), default=list(METHOD_ORDER))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if len(args.methods) != len(set(args.methods)):
        parser.error("--methods contains duplicates")
    if args.output_root is None:
        repo_root = Path(__file__).resolve().parents[2]
        results_root = Path(
            os.environ.get("SNM_RESULTS_ROOT", str(repo_root / "runs"))
        ).expanduser()
        args.output_root = results_root / "20_llama_swiglu_1b" / "capacity_exact"
    return args


def endpoint_rows(payload: dict[str, Any], method: str) -> tuple[int, int]:
    boundaries = payload.get("boundaries")
    if not isinstance(boundaries, dict) or not isinstance(boundaries.get(method), dict):
        raise RuntimeError(f"fine manifest has no boundary for {method}")
    boundary = boundaries[method]
    lower = boundary.get("max_tested_success_batch")
    upper = boundary.get("first_tested_oom_batch")
    if not isinstance(lower, int) or not isinstance(upper, int) or upper - lower != 2:
        raise RuntimeError(
            f"{method} is not eligible for odd confirmation: success={lower!r}, OOM={upper!r}"
        )

    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("fine manifest has no result rows")
    success_matches = [
        row for row in rows
        if isinstance(row, dict)
        and row.get("method") == method
        and row.get("device_batch_size") == lower
        and row.get("status") == "completed"
        and row.get("completed_steps") == 34
    ]
    oom_matches = [
        row for row in rows
        if isinstance(row, dict)
        and row.get("method") == method
        and row.get("device_batch_size") == upper
        and row.get("status") == "failed"
        and row.get("failure_class") == "oom"
    ]
    if len(success_matches) != 1 or len(oom_matches) != 1:
        raise RuntimeError(
            f"{method} parent endpoints are not uniquely certified: "
            f"success_rows={len(success_matches)}, oom_rows={len(oom_matches)}"
        )
    return lower, upper


def validate_parent(args: argparse.Namespace) -> tuple[Path, dict[str, Any], dict[str, tuple[int, int]]]:
    path = args.fine_manifest.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"fine-capacity manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else {}
    if payload.get("status") != "completed":
        failures.append(f"parent status is {payload.get('status')!r}, expected 'completed'")
    if payload.get("evidence_class") != "capacity_only":
        failures.append("parent evidence class is not capacity_only")
    if plan.get("protocol") != "microbatch_capacity_fine_fixed_accumulation_v1":
        failures.append("parent protocol is not the frozen fine-capacity protocol")
    if plan.get("accumulation_steps") != fine.ACCUMULATION_STEPS:
        failures.append("parent accumulation steps differ from 8")
    if plan.get("steps_per_cell") != 34:
        failures.append("parent cells did not use the required 34 updates")
    if plan.get("seed") != args.seed:
        failures.append(f"parent seed={plan.get('seed')!r} differs from requested seed={args.seed}")
    if Path(str(payload.get("official_repo", ""))).resolve() != args.official_repo.resolve():
        failures.append("official repository path differs from the parent fine sweep")
    if Path(str(payload.get("python_exe", ""))).resolve() != args.python_exe.resolve():
        failures.append("training Python path differs from the parent fine sweep")
    expected_hashes = {
        "fine_controller_sha256": fine.coarse.sha256_file(Path(fine.__file__)),
        "fine_worker_sha256": fine.coarse.sha256_file(fine.WORKER),
        "capacity_trainer_sha256": fine.coarse.sha256_file(fine.CAPACITY_TRAINER),
    }
    for key, expected in expected_hashes.items():
        if payload.get(key) != expected:
            failures.append(f"{key} differs from the parent fine sweep")
    if failures:
        raise RuntimeError("exact-boundary parent validation failed:\n- " + "\n- ".join(failures))

    endpoints = {method: endpoint_rows(payload, method) for method in args.methods}
    return path, payload, endpoints


def exact_boundary(lower: int, upper: int, row: dict[str, Any]) -> dict[str, Any]:
    middle = lower + 1
    if row.get("status") == "completed":
        return {
            "max_success_device_batch": middle,
            "first_oom_device_batch": upper,
            "capacity_interval": f"[{middle}, {upper})",
            "resolved_width": 1,
            "odd_confirmation": "completed",
        }
    if row.get("failure_class") == "oom":
        return {
            "max_success_device_batch": lower,
            "first_oom_device_batch": middle,
            "capacity_interval": f"[{lower}, {middle})",
            "resolved_width": 1,
            "odd_confirmation": "oom",
        }
    raise RuntimeError(f"odd confirmation has an invalid outcome: {row}")


def main() -> None:
    args = parse_args()
    parent_path, _parent, endpoints = validate_parent(args)
    cells = {
        method: {"parent_success": lower, "test_batch": lower + 1, "parent_oom": upper}
        for method, (lower, upper) in endpoints.items()
    }
    plan = {
        "protocol": PROTOCOL,
        "evidence_class": "capacity_only",
        "parent_fine_manifest": str(parent_path),
        "methods": args.methods,
        "cells": cells,
        "seed": args.seed,
        "steps_per_cell": 34,
        "sequence_length": 1024,
        "accumulation_steps": fine.ACCUMULATION_STEPS,
        "global_batch_rule": "device_batch_size * accumulation_steps",
        "selection_rule": "the unique odd batch between predeclared success/OOM endpoints",
        "quality_comparable": False,
        "timing_eligible": False,
    }
    print("1B exact odd-boundary confirmation plan")
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return

    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S+0000") + f"_capacity_exact_seed{args.seed}"
    batch_root = args.output_root.resolve() / batch_id
    batch_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "status": "running",
        "evidence_class": "capacity_only",
        "created_at": fine.coarse.now_iso(),
        "batch_id": batch_id,
        "plan": plan,
        "official_repo": str(args.official_repo.resolve()),
        "python_exe": str(args.python_exe.resolve()),
        "parent_fine_manifest_sha256": fine.coarse.sha256_file(parent_path),
        "exact_controller_sha256": fine.coarse.sha256_file(Path(__file__)),
        "fine_worker_sha256": fine.coarse.sha256_file(fine.WORKER),
        "capacity_trainer_sha256": fine.coarse.sha256_file(fine.CAPACITY_TRAINER),
        "rows": rows,
        "exact_boundaries": {},
    }
    manifest_path = batch_root / "capacity_exact_manifest.json"
    results_path = batch_root / "capacity_exact_results.csv"
    boundaries_path = batch_root / "capacity_exact_boundaries.json"
    fine.coarse.atomic_json(manifest_path, manifest)

    for method in args.methods:
        lower, upper = endpoints[method]
        row = fine.run_cell(args, batch_root, method, lower + 1)
        row["source"] = "capacity_exact_odd_confirmation"
        row["parent_fine_manifest"] = str(parent_path)
        rows.append(row)
        manifest["rows"] = rows
        manifest["last_updated_at"] = fine.coarse.now_iso()
        if row.get("status") != "completed" and row.get("failure_class") != "oom":
            manifest["status"] = "failed"
            manifest["fatal_cell"] = row
            fine.coarse.atomic_json(manifest_path, manifest)
            fine.coarse.write_csv(results_path, rows)
            raise RuntimeError(
                f"non-OOM exact-boundary failure for {method} batch {lower + 1}; see {row.get('error')}"
            )
        manifest["exact_boundaries"][method] = exact_boundary(lower, upper, row)
        fine.coarse.atomic_json(manifest_path, manifest)
        fine.coarse.write_csv(results_path, rows)

    manifest["status"] = "completed"
    manifest["completed_at"] = fine.coarse.now_iso()
    fine.coarse.atomic_json(manifest_path, manifest)
    fine.coarse.atomic_json(boundaries_path, manifest["exact_boundaries"])
    fine.coarse.write_csv(results_path, rows)
    print(f"Exact-capacity artifacts:  {batch_root}")
    print(f"Exact-capacity manifest:   {manifest_path}")
    print(f"Exact-capacity boundaries: {boundaries_path}")


if __name__ == "__main__":
    main()
