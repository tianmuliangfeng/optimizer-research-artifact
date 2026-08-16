#!/usr/bin/env python3
"""Staged, resumable orchestration for Experiment 50."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-08-14.1"
SEEDS = (2024, 2025, 2026)
STAGES = ("preflight", "pilot", "formal", "verify", "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--python-exe", required=True)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(args.gpus) != len(set(args.gpus)) or not args.gpus:
        parser.error("--gpus must be a nonempty unique list")
    return args


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def controller_python() -> str:
    return sys.executable


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def append_command(run_dir: Path, payload: dict[str, Any]) -> None:
    path = run_dir / "commands.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def copy_source_tree(source: Path, target: Path) -> None:
    if target.exists():
        return
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


def ensure_source_snapshot(args: argparse.Namespace) -> Path:
    snapshot = args.run_dir / "source_snapshot"
    manifest_path = snapshot / "source_snapshot_manifest.json"
    if manifest_path.is_file():
        payload = read_json(manifest_path)
        if payload.get("passed") is not True:
            raise RuntimeError(f"invalid existing source snapshot: {manifest_path}")
        for item in payload["files"]:
            path = snapshot / item["path"]
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raise RuntimeError(f"source snapshot drift: {path}")
        return snapshot

    scripts = args.repo / "scripts"
    for name in (
        "14_official_newton_muon_r0",
        "15_official_newton_muon_r1",
        "50_r1_global_activation_diag",
        "_shared",
    ):
        copy_source_tree(scripts / name, snapshot / "scripts" / name)
    wrapper = (
        args.repo
        / "commands/50_r1_global_activation_diag/20260814_ex50_r1_global_activation_diag.sh"
    )
    if not wrapper.is_file():
        raise RuntimeError(f"missing Experiment-50 command wrapper: {wrapper}")
    (snapshot / "commands").mkdir(parents=True, exist_ok=True)
    shutil.copy2(wrapper, snapshot / "commands" / wrapper.name)

    files = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_file() and path != manifest_path:
            files.append(
                {
                    "path": path.relative_to(snapshot).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "experiment_id": "50_r1_global_activation_diag",
            "passed": True,
            "file_count": len(files),
            "files": files,
            "created_at": datetime.now().astimezone().isoformat(),
        },
    )
    return snapshot


def frozen_paths(args: argparse.Namespace) -> dict[str, Path]:
    snapshot = ensure_source_snapshot(args)
    root = snapshot / "scripts/50_r1_global_activation_diag"
    return {
        "worker": root / "run_global_diag.py",
        "analyzer": root / "analyze_global_diag.py",
        "contract": root / "global_diag_contract.json",
        "controls": root / "frozen_r1_controls.csv",
    }


def accepted_batch(stage_dir: Path, *, pilot: bool) -> Path | None:
    accepted: list[Path] = []
    valid_statuses = (
        {"completed_valid_smoke"}
        if pilot
        else {"completed_valid", "completed_valid_local_wandb_incomplete"}
    )
    for path in sorted(stage_dir.glob("**/r1_manifest.json")):
        payload = read_json(path)
        if (
            payload.get("status") in valid_statuses
            and not payload.get("failures")
            and len(payload.get("summaries", [])) == 1
        ):
            accepted.append(path)
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
    pilot_manifest: Path | None = None,
) -> list[str]:
    paths = frozen_paths(args)
    stage_dir = args.run_dir / stage / f"seed{seed}"
    command = [
        controller_python(),
        str(paths["worker"]),
        "--official-repo",
        str(args.official_repo),
        "--python-exe",
        args.python_exe,
        "--seed",
        str(seed),
        "--methods",
        "global_diag",
        "--results-dir",
        str(stage_dir),
        "--run-prefix",
        "mainconf_r1_global_diag",
        "--continue-on-error",
    ]
    previous = resumable_batch(stage_dir)
    if previous is not None:
        command.extend(["--resume-batch", str(previous)])
    if stage == "pilot":
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
        if pilot_manifest is None:
            raise RuntimeError(f"seed {seed}: formal run requires a pilot manifest")
        command.extend(
            [
                "--smoke-manifest",
                str(pilot_manifest),
                "--wandb-mode",
                args.wandb_mode,
            ]
        )
        if args.wandb_project:
            command.extend(["--wandb-project", args.wandb_project])
        if args.wandb_entity:
            command.extend(["--wandb-entity", args.wandb_entity])
    return command


def run_jobs(args: argparse.Namespace, jobs: list[dict[str, Any]]) -> None:
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
            handle = log_path.open("a", encoding="utf-8", buffering=1)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["NANOGPT_WORKSPACE_ROOT"] = str(args.repo.parent)
            env["SELECTIVE_NEWTON_MUON_MAIN_CONFERENCE_REPO"] = str(args.repo)
            command = list(job["command"])
            append_command(
                args.run_dir,
                {
                    "label": label,
                    "gpu": gpu,
                    "command": command,
                    "command_text": command_text(command),
                    "log": str(log_path),
                    "started_at": datetime.now().astimezone().isoformat(),
                },
            )
            print(f"START gpu={gpu} label={label}", flush=True)
            process = subprocess.Popen(
                command,
                cwd=args.repo,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active[gpu] = {
                **job,
                "process": process,
                "handle": handle,
                "log": log_path,
            }
        finished: list[str] = []
        for gpu, job in active.items():
            code = job["process"].poll()
            if code is None:
                continue
            job["handle"].close()
            print(f"END gpu={gpu} label={job['label']} return_code={code}", flush=True)
            if code != 0:
                failures.append(
                    {
                        "label": job["label"],
                        "gpu": gpu,
                        "return_code": code,
                        "log": str(job["log"]),
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
                job["handle"].close()
            write_json(args.run_dir / "worker_failures.json", failures)
            raise RuntimeError(f"Experiment-50 worker failure: {failures}")
        if pending or active:
            time.sleep(2)


def require_pass(path: Path, label: str) -> None:
    if not path.is_file() or read_json(path).get("passed") is not True:
        raise RuntimeError(f"{label} has not passed: {path}")


def run_preflight(args: argparse.Namespace) -> None:
    status = args.run_dir / "preflight_manifest.json"
    if status.is_file() and read_json(status).get("passed") is True:
        print("skip passed preflight")
        return
    paths = frozen_paths(args)
    log = args.run_dir / "controller_logs/preflight.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        controller_python(),
        str(paths["worker"]),
        "--official-repo",
        str(args.official_repo),
        "--python-exe",
        args.python_exe,
        "--seed",
        "2024",
        "--methods",
        "global_diag",
        "--preflight",
        "--wandb-mode",
        "disabled",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus[0]
    env["NANOGPT_WORKSPACE_ROOT"] = str(args.repo.parent)
    env["SELECTIVE_NEWTON_MUON_MAIN_CONFERENCE_REPO"] = str(args.repo)
    append_command(
        args.run_dir,
        {
            "label": "preflight",
            "gpu": args.gpus[0],
            "command": command,
            "command_text": command_text(command),
            "log": str(log),
        },
    )
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=args.repo,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    payload = {
        "schema_version": 1,
        "experiment_id": "50_r1_global_activation_diag",
        "passed": completed.returncode == 0,
        "return_code": completed.returncode,
        "gpu": args.gpus[0],
        "contract_sha256": sha256_file(paths["contract"]),
        "source_snapshot_manifest_sha256": sha256_file(
            args.run_dir / "source_snapshot/source_snapshot_manifest.json"
        ),
        "log": str(log),
    }
    write_json(status, payload)
    if not payload["passed"]:
        raise RuntimeError(f"Experiment-50 preflight failed; inspect {log}")


def run_pilot(args: argparse.Namespace) -> None:
    require_pass(args.run_dir / "preflight_manifest.json", "preflight")
    jobs: list[dict[str, Any]] = []
    for seed in SEEDS:
        stage_dir = args.run_dir / "pilot" / f"seed{seed}"
        if accepted_batch(stage_dir, pilot=True) is None:
            jobs.append(
                {
                    "label": f"pilot/seed{seed}",
                    "command": runner_command(args, seed=seed, stage="pilot"),
                }
            )
    run_jobs(args, jobs)
    accepted = {
        seed: accepted_batch(args.run_dir / "pilot" / f"seed{seed}", pilot=True)
        for seed in SEEDS
    }
    if any(path is None for path in accepted.values()):
        raise RuntimeError(f"Experiment-50 pilot incomplete: {accepted}")
    write_json(
        args.run_dir / "pilot_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": "50_r1_global_activation_diag",
            "passed": True,
            "seeds": list(SEEDS),
            "steps": 34,
            "crosses_first_refresh": True,
            "accepted_batches": {seed: str(path) for seed, path in accepted.items()},
        },
    )


def run_formal(args: argparse.Namespace) -> None:
    require_pass(args.run_dir / "pilot_manifest.json", "pilot")
    jobs: list[dict[str, Any]] = []
    for seed in SEEDS:
        stage_dir = args.run_dir / "formal" / f"seed{seed}"
        if accepted_batch(stage_dir, pilot=False) is not None:
            print(f"skip accepted formal seed {seed}")
            continue
        pilot = accepted_batch(
            args.run_dir / "pilot" / f"seed{seed}", pilot=True
        )
        if pilot is None:
            raise RuntimeError(f"seed {seed}: accepted pilot vanished")
        jobs.append(
            {
                "label": f"formal/seed{seed}",
                "command": runner_command(
                    args,
                    seed=seed,
                    stage="formal",
                    pilot_manifest=pilot,
                ),
            }
        )
    run_jobs(args, jobs)
    accepted = {
        seed: accepted_batch(args.run_dir / "formal" / f"seed{seed}", pilot=False)
        for seed in SEEDS
    }
    if any(path is None for path in accepted.values()):
        raise RuntimeError(f"Experiment-50 formal incomplete: {accepted}")
    write_json(
        args.run_dir / "formal_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": "50_r1_global_activation_diag",
            "passed": True,
            "formal_units": 3,
            "seeds": list(SEEDS),
            "accepted_batches": {seed: str(path) for seed, path in accepted.items()},
            "wandb_required_for_scientific_validity": False,
            "timing_usable": False,
        },
    )


def run_verify(args: argparse.Namespace) -> None:
    require_pass(args.run_dir / "formal_manifest.json", "formal")
    paths = frozen_paths(args)
    output = args.run_dir / "analysis"
    manifest = output / "analysis_manifest.json"
    if manifest.is_file() and read_json(manifest).get("passed") is True:
        print("skip passed analysis")
    else:
        command = [
            controller_python(),
            str(paths["analyzer"]),
            "--run-dir",
            str(args.run_dir),
            "--contract",
            str(paths["contract"]),
            "--controls",
            str(paths["controls"]),
            "--output-dir",
            str(output),
        ]
        append_command(
            args.run_dir,
            {
                "label": "verify",
                "command": command,
                "command_text": command_text(command),
            },
        )
        subprocess.run(command, cwd=args.repo, check=True)
    require_pass(manifest, "analysis")
    analysis = read_json(manifest)
    write_json(
        args.run_dir / "handoff_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": "50_r1_global_activation_diag",
            "status": "completed",
            "passed": True,
            "scientific_result": analysis["classification"],
            "analysis_manifest": str(manifest),
            "analysis_manifest_sha256": sha256_file(manifest),
            "formal_units": 3,
            "timing_usable": False,
            "completed_at": datetime.now().astimezone().isoformat(),
        },
    )
    print("Experiment 50 completed.")
    print(f"Artifacts: {args.run_dir}")
    print(f"Analysis: {manifest}")
    print(f"Scientific result: {analysis['classification']}")


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.repo = args.repo.expanduser().resolve()
    args.official_repo = args.official_repo.expanduser().resolve()
    if args.run_dir.exists() and any(args.run_dir.iterdir()) and not args.resume:
        raise RuntimeError(
            f"run directory is nonempty; pass --resume to continue: {args.run_dir}"
        )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    paths = frozen_paths(args)
    contract = read_json(paths["contract"])
    if args.gpus != contract["execution_policy"]["physical_gpus"]:
        raise RuntimeError(
            "Experiment 50 is frozen to physical GPUs 0 and 1: "
            f"expected={contract['execution_policy']['physical_gpus']} "
            f"observed={args.gpus}"
        )
    write_json(
        args.run_dir / "suite_plan.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "experiment_id": "50_r1_global_activation_diag",
            "stage_requested": args.stage,
            "run_dir": str(args.run_dir),
            "repo": str(args.repo),
            "official_repo": str(args.official_repo),
            "controller_python": controller_python(),
            "training_python": args.python_exe,
            "gpus": args.gpus,
            "seeds": list(SEEDS),
            "methods": ["global_diag"],
            "contract_sha256": sha256_file(paths["contract"]),
            "controls_sha256": sha256_file(paths["controls"]),
            "wandb_mode": args.wandb_mode,
            "timing_usable": False,
        },
    )

    stages = (
        ("preflight", "pilot", "formal", "verify")
        if args.stage == "all"
        else (args.stage,)
    )
    dispatch = {
        "preflight": run_preflight,
        "pilot": run_pilot,
        "formal": run_formal,
        "verify": run_verify,
    }
    for stage in stages:
        dispatch[stage](args)


if __name__ == "__main__":
    main()
