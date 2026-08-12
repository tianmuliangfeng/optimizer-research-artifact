#!/usr/bin/env python3
"""Orchestrate smoke, six formal runs, W&B upload, and experiment-41 analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-07-30.2"
SCRIPT_DIR = Path(__file__).resolve().parent
SEEDS = (2024, 2025, 2026)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--python-exe", required=True)
    parser.add_argument("--existing-summary", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(args.gpus) < 1 or len(args.gpus) != len(set(args.gpus)):
        parser.error("--gpus must contain unique device identifiers")
    return args


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def controller_python() -> str:
    """Preserve the venv entrypoint instead of resolving its interpreter symlink."""
    return sys.executable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def accepted_batch(stage_dir: Path, formal: bool) -> Path | None:
    accepted: list[Path] = []
    for manifest_path in sorted(stage_dir.glob("**/r1_manifest.json")):
        payload = read_json(manifest_path)
        expected = "completed_valid" if formal else "completed_valid_smoke"
        if payload.get("status") == expected and not payload.get("failures"):
            if formal and payload.get("wandb_complete") is not True:
                continue
            accepted.append(manifest_path)
    if len(accepted) > 1:
        raise RuntimeError(f"multiple accepted batches under {stage_dir}: {accepted}")
    return accepted[0] if accepted else None


def resumable_batch(stage_dir: Path) -> Path | None:
    plans = sorted(
        stage_dir.glob("**/r1_plan.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return plans[0].parent if plans else None


def runner_command(
    args: argparse.Namespace,
    *,
    seed: int,
    stage: str,
    smoke_manifest: Path | None = None,
) -> list[str]:
    stage_dir = args.run_dir / stage / f"seed{seed}"
    command = [
        controller_python(),
        str(
            args.repo
            / "scripts/41_r1_kstate_module_factorial/run_r1_module_factorial.py"
        ),
        "--official-repo",
        str(args.official_repo),
        "--python-exe",
        args.python_exe,
        "--seed",
        str(seed),
        "--methods",
        "block4",
        "none",
        "--results-dir",
        str(stage_dir),
        "--run-prefix",
        "mainconf_r1_kstate_factorial_cfc_none",
        "--continue-on-error",
    ]
    previous = resumable_batch(stage_dir)
    if previous is not None:
        command.extend(["--resume-batch", str(previous)])
    if stage == "smoke":
        command.extend(
            [
                "--numerical-smoke",
                "--smoke-steps",
                "34",
                "--wandb-mode",
                "disabled",
            ]
        )
    else:
        if smoke_manifest is None:
            raise RuntimeError(f"seed {seed}: formal job requires smoke manifest")
        command.extend(
            [
                "--smoke-manifest",
                str(smoke_manifest),
                "--wandb-mode",
                "online",
            ]
        )
        if args.wandb_project:
            command.extend(["--wandb-project", args.wandb_project])
        if args.wandb_entity:
            command.extend(["--wandb-entity", args.wandb_entity])
    return command


def append_command(run_dir: Path, record: dict[str, Any]) -> None:
    path = run_dir / "commands.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_parallel_jobs(
    args: argparse.Namespace,
    jobs: list[dict[str, Any]],
) -> None:
    pending = list(jobs)
    active: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    logs = args.run_dir / "controller_logs"
    logs.mkdir(parents=True, exist_ok=True)
    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending:
                continue
            job = pending.pop(0)
            label = str(job["label"])
            log_path = logs / f"{label.replace('/', '_')}.log"
            log_handle = log_path.open("a", encoding="utf-8", buffering=1)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            command = list(job["command"])
            print(f"START gpu={gpu} label={label}")
            append_command(
                args.run_dir,
                {
                    "label": label,
                    "gpu": gpu,
                    "command": command,
                    "command_text": command_text(command),
                    "started_at": datetime.now().astimezone().isoformat(),
                    "log": str(log_path),
                },
            )
            process = subprocess.Popen(
                command,
                cwd=args.repo,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active[gpu] = {
                **job,
                "process": process,
                "log_handle": log_handle,
                "log_path": log_path,
            }
        finished: list[str] = []
        for gpu, job in active.items():
            return_code = job["process"].poll()
            if return_code is None:
                continue
            job["log_handle"].close()
            label = str(job["label"])
            print(f"END gpu={gpu} label={label} return_code={return_code}")
            if return_code != 0:
                failures.append(
                    {
                        "label": label,
                        "gpu": gpu,
                        "return_code": return_code,
                        "log": str(job["log_path"]),
                        "command": job["command"],
                    }
                )
            finished.append(gpu)
        for gpu in finished:
            del active[gpu]
        if failures:
            for job in active.values():
                job["process"].terminate()
            for job in active.values():
                try:
                    job["process"].wait(timeout=30)
                except subprocess.TimeoutExpired:
                    job["process"].kill()
                job["log_handle"].close()
            write_json(args.run_dir / "worker_failures.json", failures)
            raise RuntimeError(f"experiment-41 worker failures: {failures}")
        if pending or active:
            time.sleep(10)


def run_preflight(args: argparse.Namespace) -> None:
    status_path = args.run_dir / "preflight.json"
    if status_path.is_file() and read_json(status_path).get("passed") is True:
        return
    log_path = args.run_dir / "controller_logs/preflight.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        controller_python(),
        str(
            args.repo
            / "scripts/41_r1_kstate_module_factorial/run_r1_module_factorial.py"
        ),
        "--official-repo",
        str(args.official_repo),
        "--python-exe",
        args.python_exe,
        "--seed",
        "2026",
        "--methods",
        "block4",
        "none",
        "--preflight",
        "--wandb-mode",
        "disabled",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus[0]
    append_command(
        args.run_dir,
        {
            "label": "preflight",
            "gpu": args.gpus[0],
            "command": command,
            "command_text": command_text(command),
            "log": str(log_path),
        },
    )
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=args.repo,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    write_json(
        status_path,
        {
            "passed": completed.returncode == 0,
            "return_code": completed.returncode,
            "gpu": args.gpus[0],
            "log": str(log_path),
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(f"experiment-41 preflight failed; inspect {log_path}")


def analysis_dir(run_dir: Path) -> Path:
    primary = run_dir / "analysis"
    candidates = [primary, *sorted(run_dir.glob("analysis_retry_*"))]
    for candidate in candidates:
        manifest = candidate / "r1_module_factorial_analysis_manifest.json"
        if manifest.is_file() and read_json(manifest).get("passed") is True:
            return candidate
    if not primary.exists():
        return primary
    index = 1
    while (run_dir / f"analysis_retry_{index}").exists():
        index += 1
    return run_dir / f"analysis_retry_{index}"


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.repo = args.repo.expanduser().resolve()
    args.official_repo = args.official_repo.expanduser().resolve()
    args.existing_summary = args.existing_summary.expanduser().resolve()
    if args.run_dir.exists() and any(args.run_dir.iterdir()) and not args.resume:
        raise RuntimeError(
            f"run directory is nonempty; pass --resume to continue: {args.run_dir}"
        )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    contract = (
        args.repo
        / "scripts/41_r1_kstate_module_factorial/factorial_contract.json"
    )
    contract_payload = read_json(contract)
    expected_gpus = contract_payload["execution_policy"]["physical_gpus"]
    if args.gpus != expected_gpus:
        raise RuntimeError(
            f"experiment 41 is frozen to physical GPUs 0 and 1: "
            f"expected {expected_gpus}, observed {args.gpus}"
        )
    suite_plan = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "status": "running",
        "run_dir": str(args.run_dir),
        "repo": str(args.repo),
        "official_repo": str(args.official_repo),
        "controller_python": controller_python(),
        "training_python": args.python_exe,
        "python_exe": args.python_exe,
        "gpus": args.gpus,
        "seeds": list(SEEDS),
        "methods": ["block4", "none"],
        "new_training_cells": ["cproj_only", "neither"],
        "reused_cells": ["both", "fc_only"],
        "contract_sha256": sha256_file(contract),
        "existing_summary": str(args.existing_summary),
        "existing_summary_sha256": sha256_file(args.existing_summary),
        "timing_usable": False,
    }
    write_json(args.run_dir / "suite_plan.json", suite_plan)

    run_preflight(args)

    smoke_jobs: list[dict[str, Any]] = []
    smoke_manifests: dict[int, Path] = {}
    for seed in SEEDS:
        stage_dir = args.run_dir / "smoke" / f"seed{seed}"
        accepted = accepted_batch(stage_dir, formal=False)
        if accepted is not None:
            smoke_manifests[seed] = accepted
        else:
            smoke_jobs.append(
                {
                    "label": f"smoke/seed{seed}",
                    "command": runner_command(
                        args, seed=seed, stage="smoke"
                    ),
                }
            )
    run_parallel_jobs(args, smoke_jobs)
    for seed in SEEDS:
        accepted = accepted_batch(
            args.run_dir / "smoke" / f"seed{seed}", formal=False
        )
        if accepted is None:
            raise RuntimeError(f"seed {seed}: accepted smoke missing after run")
        smoke_manifests[seed] = accepted

    formal_jobs: list[dict[str, Any]] = []
    for seed in SEEDS:
        stage_dir = args.run_dir / "formal" / f"seed{seed}"
        if accepted_batch(stage_dir, formal=True) is None:
            formal_jobs.append(
                {
                    "label": f"formal/seed{seed}",
                    "command": runner_command(
                        args,
                        seed=seed,
                        stage="formal",
                        smoke_manifest=smoke_manifests[seed],
                    ),
                }
            )
    run_parallel_jobs(args, formal_jobs)
    for seed in SEEDS:
        if (
            accepted_batch(
                args.run_dir / "formal" / f"seed{seed}", formal=True
            )
            is None
        ):
            raise RuntimeError(f"seed {seed}: accepted formal batch missing")

    output_dir = analysis_dir(args.run_dir)
    analysis_manifest = output_dir / "r1_module_factorial_analysis_manifest.json"
    if not (
        analysis_manifest.is_file()
        and read_json(analysis_manifest).get("passed") is True
    ):
        command = [
            controller_python(),
            str(
                args.repo
                / "scripts/41_r1_kstate_module_factorial/analyze_r1_module_factorial.py"
            ),
            "--run-dir",
            str(args.run_dir),
            "--existing-summary",
            str(args.existing_summary),
            "--contract",
            str(contract),
            "--output-dir",
            str(output_dir),
        ]
        append_command(
            args.run_dir,
            {
                "label": "analysis",
                "command": command,
                "command_text": command_text(command),
            },
        )
        subprocess.run(command, cwd=args.repo, check=True)

    analysis = read_json(analysis_manifest)
    final = {
        **suite_plan,
        "status": "completed",
        "passed": analysis["passed"],
        "classification": analysis["classification"],
        "material_interaction": analysis["material_interaction"],
        "analysis_manifest": str(analysis_manifest),
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    write_json(args.run_dir / "r1_module_factorial_manifest.json", final)
    resume = (
        f"{shlex.quote(controller_python())} "
        f"{shlex.quote(str(args.repo / 'scripts/41_r1_kstate_module_factorial/run_r1_module_factorial_suite.py'))} "
        f"--run-dir {shlex.quote(str(args.run_dir))} "
        f"--repo {shlex.quote(str(args.repo))} "
        f"--official-repo {shlex.quote(str(args.official_repo))} "
        f"--python-exe {shlex.quote(args.python_exe)} "
        f"--existing-summary {shlex.quote(str(args.existing_summary))} "
        f"--gpus {' '.join(map(shlex.quote, args.gpus))} --resume"
    )
    print(f"R1 module factorial artifacts: {args.run_dir}")
    print(f"R1 module factorial manifest: {args.run_dir / 'r1_module_factorial_manifest.json'}")
    print(f"RESUME_COMMAND={resume}")


if __name__ == "__main__":
    main()
