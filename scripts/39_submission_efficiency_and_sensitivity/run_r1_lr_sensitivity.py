#!/usr/bin/env python3
"""Recoverable lane controller for the R1 shared LR multiplier grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-07-29.4"
FAMILY = "39_r1_shared_lr_sensitivity"
SMOKE_PROTOCOL = "r1_shared_recipe_lr_multiplier_exact_shape_smoke_v1"
FORMAL_PROTOCOL = "r1_shared_recipe_lr_multiplier_supporting_v1"
SMOKE_STEPS = 34
BASE_LR = {"muon": 0.0036, "block4": 0.004, "none": 0.004, "diag": 0.004}
MATRIX_LR = {
    "muon": 0.00036,
    "block4": 0.0004,
    "none": 0.0004,
    "diag": 0.0004,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--training-python", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument(
        "--multipliers", nargs="+", type=float, default=[0.8, 1.0, 1.2]
    )
    parser.add_argument("--budget-steps", type=int, default=3000)
    parser.add_argument("--warmdown-steps", type=int, default=871)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-entity")
    args = parser.parse_args()
    if set(args.multipliers) != {0.8, 1.0, 1.2}:
        parser.error("--multipliers must contain exactly 0.8 1.0 1.2")
    if set(args.methods) - {"diag", "none", "block4", "muon"}:
        parser.error("unknown method")
    return args


def multiplier_label(value: float) -> str:
    return f"m{value:g}".replace(".", "p")


def valid_manifest(
    path: Path,
    smoke: bool,
    methods: list[str],
    *,
    seed: int,
    budget_steps: int,
    multiplier: float,
) -> bool:
    if not path.is_file():
        return False
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    expected_status = "completed_valid_smoke" if smoke else "completed_valid"
    expected_protocol = SMOKE_PROTOCOL if smoke else FORMAL_PROTOCOL
    expected_final_step = SMOKE_STEPS if smoke else budget_steps
    expected_profile = (
        "exact_shape_numerical_smoke"
        if smoke
        else "shared_recipe_lr_sensitivity_supporting"
    )
    summaries = value.get("summaries", ())
    if not isinstance(summaries, list):
        return False
    return (
        value.get("status") == expected_status
        and value.get("family") == FAMILY
        and value.get("protocol") == expected_protocol
        and value.get("methods") == methods
        and value.get("seed") == seed
        and value.get("formal_evidence") is False
        and value.get("evidence_profile") == expected_profile
        and len(summaries) == len(methods)
        and {summary.get("method") for summary in summaries} == set(methods)
        and all(summary.get("controlled_seed") == seed for summary in summaries)
        and all(
            summary.get("final_val_step") == expected_final_step
            for summary in summaries
        )
        and all(
            isinstance(summary.get("base_learning_rate"), (int, float))
            and math.isclose(
                float(summary["base_learning_rate"]),
                BASE_LR[summary["method"]] * multiplier,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            and isinstance(summary.get("matrix_learning_rate"), (int, float))
            and math.isclose(
                float(summary["matrix_learning_rate"]),
                MATRIX_LR[summary["method"]] * multiplier,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            for summary in summaries
        )
        and (smoke or value.get("wandb_complete") is True)
    )


def batch_candidates(cell: Path, smoke: bool, seed: int) -> list[Path]:
    kind = "smoke" if smoke else "formal"
    return sorted(cell.glob(f"*_{kind}_seed{seed}"), reverse=True)


def locate_valid_batch(
    cell: Path,
    smoke: bool,
    methods: list[str],
    *,
    seed: int,
    budget_steps: int,
    multiplier: float,
) -> Path | None:
    return next(
        (
            candidate
            for candidate in batch_candidates(cell, smoke, seed)
            if valid_manifest(
                candidate / "r1_manifest.json",
                smoke,
                methods,
                seed=seed,
                budget_steps=budget_steps,
                multiplier=multiplier,
            )
        ),
        None,
    )


def locate_resumable_batch(cell: Path, smoke: bool, seed: int) -> Path | None:
    return next(
        (
            candidate
            for candidate in batch_candidates(cell, smoke, seed)
            if (candidate / "r1_plan.json").is_file()
        ),
        None,
    )


def run_phase(
    args: argparse.Namespace,
    worker: Path,
    cell: Path,
    multiplier: float,
    *,
    smoke: bool,
    smoke_manifest: Path | None = None,
) -> Path:
    completed = locate_valid_batch(
        cell,
        smoke,
        args.methods,
        seed=args.seed,
        budget_steps=args.budget_steps,
        multiplier=multiplier,
    )
    if completed is not None:
        return completed
    existing = locate_resumable_batch(cell, smoke, args.seed)
    command = [
        sys.executable,
        str(worker),
        "--repo",
        str(args.repo),
        "--official-repo",
        str(args.official_repo),
        "--python-exe",
        str(args.training_python),
        "--results-dir",
        str(cell),
        "--methods",
        *args.methods,
        "--lr-multiplier",
        str(multiplier),
        "--budget-steps",
        str(args.budget_steps),
        "--warmdown-steps",
        str(args.warmdown_steps),
        "--seed",
        str(args.seed),
        "--wandb-project",
        args.wandb_project,
    ]
    if args.wandb_entity:
        command.extend(["--wandb-entity", args.wandb_entity])
    if smoke:
        command.extend(["--smoke", "--smoke-steps", str(SMOKE_STEPS)])
    else:
        assert smoke_manifest is not None
        command.extend(["--smoke-manifest", str(smoke_manifest)])
    if existing is not None and (existing / "r1_plan.json").is_file():
        existing_manifest_path = existing / "r1_manifest.json"
        existing_status = ""
        if existing_manifest_path.is_file():
            try:
                existing_status = read_json(existing_manifest_path).get(
                    "status", ""
                )
            except (OSError, json.JSONDecodeError):
                existing_status = ""
        # The reused R1 controller can resume failed methods, but its legacy
        # upload retry path is intentionally limited to formal-evidence runs.
        # A locally complete supporting run with an interrupted W&B upload is
        # therefore rerun in a new immutable batch.
        if smoke or existing_status not in {
            "completed_valid_local_wandb_incomplete",
            "completed_valid_local",
        }:
            command.extend(["--resume-batch", str(existing)])
    log_path = cell / ("controller_smoke.log" if smoke else "controller_formal.log")
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND " + json.dumps(command) + "\n")
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"{'smoke' if smoke else 'formal'} failed for multiplier={multiplier}; "
            f"see {log_path}"
        )
    batch = locate_valid_batch(
        cell,
        smoke,
        args.methods,
        seed=args.seed,
        budget_steps=args.budget_steps,
        multiplier=multiplier,
    )
    if batch is None:
        raise RuntimeError(f"completed phase has no valid manifest: {cell}")
    return batch


def main() -> None:
    args = parse_args()
    contract = (
        args.repo
        / "scripts/39_submission_efficiency_and_sensitivity/lr_sensitivity_contract.json"
    ).resolve()
    worker = (
        args.repo
        / "scripts/39_submission_efficiency_and_sensitivity/r1_lr_sensitivity_worker.py"
    ).resolve()
    if not contract.is_file() or not worker.is_file():
        raise RuntimeError("sensitivity contract or worker is missing")
    lane_dir = args.run_dir.resolve() / args.lane
    lane_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for multiplier in args.multipliers:
        cell = lane_dir / multiplier_label(multiplier)
        cell.mkdir(parents=True, exist_ok=True)
        smoke_batch = run_phase(args, worker, cell, multiplier, smoke=True)
        formal_batch = run_phase(
            args,
            worker,
            cell,
            multiplier,
            smoke=False,
            smoke_manifest=smoke_batch / "r1_manifest.json",
        )
        entries.append(
            {
                "multiplier": multiplier,
                "smoke_batch": str(
                    smoke_batch.relative_to(args.run_dir.resolve())
                ),
                "smoke_manifest": str(
                    (smoke_batch / "r1_manifest.json").relative_to(
                        args.run_dir.resolve()
                    )
                ),
                "formal_batch": str(
                    formal_batch.relative_to(args.run_dir.resolve())
                ),
                "formal_manifest": str(
                    (formal_batch / "r1_manifest.json").relative_to(
                        args.run_dir.resolve()
                    )
                ),
            }
        )
        write_json(
            lane_dir / "lane_manifest.json",
            {
                "schema_version": 1,
                "script_version": SCRIPT_VERSION,
                "status": "running",
                "lane": args.lane,
                "methods": args.methods,
                "multipliers": args.multipliers,
                "seed": args.seed,
                "budget_steps": args.budget_steps,
                "warmdown_steps": args.warmdown_steps,
                "contract_sha256": sha256_file(contract),
                "worker_sha256": sha256_file(worker),
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "entries": entries,
            },
        )
    write_json(
        lane_dir / "lane_manifest.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "lane": args.lane,
            "methods": args.methods,
            "multipliers": args.multipliers,
            "seed": args.seed,
            "budget_steps": args.budget_steps,
            "warmdown_steps": args.warmdown_steps,
            "contract_sha256": sha256_file(contract),
            "worker_sha256": sha256_file(worker),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "entries": entries,
        },
    )
    print(f"R1 LR-sensitivity lane manifest: {lane_dir / 'lane_manifest.json'}")


if __name__ == "__main__":
    main()
